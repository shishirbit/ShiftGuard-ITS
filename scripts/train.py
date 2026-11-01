from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from shiftguard.checkpointing import restore_training_state, save_training_state
from shiftguard.registry import LEGACY_ABLATIONS, REFERENCE_ABLATIONS, shiftguard_flags
from shiftguard.metrics import horizon_metrics
from shiftguard.models import (
    AGCRNBaseline,
    DCRNNBaseline,
    GraphWaveNetBaseline,
    GRUBaseline,
    LaplaceHead,
    LinearBaseline,
    MLPBaseline,
    Persistence,
    ShiftGuard,
    STAEformerBaseline,
    STGCNBaseline,
    STIDBaseline,
    laplace_loss,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")


def load_split(root: Path, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(root / f"{name}.npz") as archive:
        return torch.from_numpy(archive["x"]), torch.from_numpy(archive["y"])


def correlation_adjacency(x: torch.Tensor, top_k: int = 12) -> torch.Tensor:
    # Use training inputs only and a subsample to control memory.
    values = x[:: max(1, len(x) // 5000), :, :, 0].reshape(-1, x.shape[2]).numpy()
    correlation = np.nan_to_num(np.corrcoef(values, rowvar=False), nan=0.0)
    np.fill_diagonal(correlation, 0.0)
    adjacency = np.zeros_like(correlation, dtype=np.float32)
    for node in range(len(correlation)):
        selected = np.argpartition(correlation[node], -top_k)[-top_k:]
        adjacency[node, selected] = np.maximum(correlation[node, selected], 0)
    return torch.from_numpy(adjacency)


BASE_MODELS = (
    "persistence",
    "linear",
    "mlp",
    "gru",
    "stgcn",
    "dcrnn",
    "graphwavenet",
    "agcrn",
    "stid",
    "staeformer",
    *dict.fromkeys(("shiftguard", *LEGACY_ABLATIONS, *REFERENCE_ABLATIONS)),
)
# Baselines retrained with ShiftGuard's structured-outage augmentation, so that
# stress comparisons separate the training recipe from the architecture.
OUTAGE_AUGMENTED_SUFFIX = "_outageaug"
OUTAGE_AUGMENTED_BASELINES = (
    "mlp",
    "gru",
    "dcrnn",
    "graphwavenet",
    "agcrn",
    "stid",
    "staeformer",
)
# Same baselines with ShiftGuard's Laplace median/scale output and loss, so
# conformal intervals and risk control are compared on equal footing.
LAPLACE_SUFFIX = "_laplace"
MODEL_CHOICES = (
    BASE_MODELS
    + tuple(f"{name}{OUTAGE_AUGMENTED_SUFFIX}" for name in OUTAGE_AUGMENTED_BASELINES)
    + tuple(
        f"{name}{OUTAGE_AUGMENTED_SUFFIX}{LAPLACE_SUFFIX}"
        for name in OUTAGE_AUGMENTED_BASELINES
    )
)


def base_model_name(name: str) -> str:
    return name.removesuffix(LAPLACE_SUFFIX).removesuffix(OUTAGE_AUGMENTED_SUFFIX)


def build_model(name: str, nodes: int, adjacency: torch.Tensor) -> nn.Module:
    if name.endswith(LAPLACE_SUFFIX):
        horizon = 12
        return LaplaceHead(build_baseline(base_model_name(name), nodes, adjacency, 2 * horizon), horizon)
    name = base_model_name(name)
    if name in OUTAGE_AUGMENTED_BASELINES:
        return build_baseline(name, nodes, adjacency)
    if name == "persistence":
        return Persistence()
    if name == "linear":
        return LinearBaseline()
    if name == "stgcn":
        return STGCNBaseline(adjacency)
    if name.startswith("shiftguard"):
        flags = shiftguard_flags(name)
        options = {}
        if "norouter" in flags:
            options["regimes"] = 1
        return ShiftGuard(
            nodes,
            adjacency,
            use_reliability="noreliability" not in flags,
            use_adaptive_adjacency="staticgraph" not in flags,
            use_static_adjacency="adaptivegraph" not in flags,
            use_mask_input="nomaskinput" not in flags,
            **options,
        )
    raise ValueError(name)


def build_baseline(
    name: str, nodes: int, adjacency: torch.Tensor, horizon: int = 12
) -> nn.Module:
    if name == "mlp":
        return MLPBaseline(horizon=horizon)
    if name == "gru":
        return GRUBaseline(horizon=horizon)
    if name == "dcrnn":
        return DCRNNBaseline(adjacency, horizon=horizon)
    if name == "graphwavenet":
        return GraphWaveNetBaseline(nodes, adjacency, horizon=horizon)
    if name == "agcrn":
        return AGCRNBaseline(nodes, horizon=horizon)
    if name == "stid":
        return STIDBaseline(nodes, horizon=horizon)
    if name == "staeformer":
        return STAEformerBaseline(nodes, horizon=horizon)
    raise ValueError(name)


def unpack(output: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, None


def backward_batch(model, x, y, observed, amp=False, microbatch_size=0):
    """Accumulate a masked loss with the same effective optimizer batch size."""
    size = microbatch_size or len(x)
    total = y[..., 1:2].sum().clamp_min(1)
    for start in range(0, len(x), size):
        stop = start + size
        target, mask = y[start:stop, ..., :1], y[start:stop, ..., 1:2]
        count = mask.sum().clamp_min(1)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=amp):
            point, scale = unpack(model(x[start:stop], observed[start:stop]))
            loss = (
                laplace_loss(point, scale, target, mask)
                if scale is not None
                else (torch.abs(point - target) * mask).sum() / count
            )
        (loss * count / total).backward()


def structured_observation_mask(x: torch.Tensor) -> torch.Tensor:
    """Mix independent, burst, and whole-node failures during training."""

    batch, history, nodes, _ = x.shape
    rate = float(torch.empty(1).uniform_(0.0, 0.4))
    family = int(torch.randint(0, 3, (1,)))
    if family == 0:
        return (torch.rand_like(x[..., :1]) >= rate).to(x.dtype)
    mask = torch.ones_like(x[..., :1])
    if family == 1:
        node_fail = torch.rand(batch, nodes, device=x.device) < rate
        start = torch.randint(0, history, (batch, nodes), device=x.device)
        steps = torch.arange(history, device=x.device)[None, :, None]
        failed = node_fail[:, None, :] & (steps >= start[:, None, :])
        return (~failed).to(x.dtype).unsqueeze(-1)
    count = max(1, round(nodes * rate))
    for sample in range(batch):
        failed = torch.randperm(nodes, device=x.device)[:count]
        mask[sample, :, failed, :] = 0
    return mask


def mask_inputs(x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Zero standardized speeds at unobserved positions; keep other channels."""

    return torch.cat([x[..., :1] * observed, x[..., 1:]], dim=-1)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets = [], []
    for x, y in loader:
        x = x.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            point, _ = unpack(model(x, torch.ones_like(x[..., :1])))
        predictions.append(point.float().cpu().numpy())
        target = y[..., :1].numpy()
        target[y[..., 1:2].numpy() == 0] = np.nan
        targets.append(target)
    return np.concatenate(predictions), np.concatenate(targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["METR-LA", "PEMS-BAY"], required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--amp", action="store_true", help="Use CUDA bfloat16 autocast.")
    parser.add_argument("--resume", action="store_true", help="Resume the last complete epoch.")
    parser.add_argument("--microbatch-size", type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = args.data_root / args.dataset
    train_x, train_y = load_split(root, "train")
    val_x, val_y = load_split(root, "val")
    adjacency = correlation_adjacency(train_x)
    model = build_model(args.model, train_x.shape[2], adjacency).to(device)
    use_amp = args.amp and device.type == "cuda"
    staeformer_cap = 32 if use_amp else 24
    effective_batch_size = (
        min(args.batch_size, staeformer_cap)
        if base_model_name(args.model) == "staeformer"
        else args.batch_size
    )
    inference_batch_size = (
        min(effective_batch_size * 2, 8)
        if base_model_name(args.model) == "staeformer"
        else effective_batch_size * 2
    )
    loaders = {
        "train": DataLoader(TensorDataset(train_x, train_y), effective_batch_size, shuffle=True),
        "val": DataLoader(TensorDataset(val_x, val_y), inference_batch_size),
    }
    output_dir = args.output_root / args.dataset / args.model / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best.pt"
    recovery = output_dir / "last_training.pt"
    config = {
        "dataset": args.dataset, "model": args.model, "seed": args.seed,
        "batch_size": effective_batch_size, "amp": use_amp, "patience": args.patience,
        "device": str(device),
    }

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    started = time.perf_counter()
    elapsed_before = 0.0
    if trainable:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, stale = float("inf"), 0
        start_epoch = 1
        if args.resume and recovery.exists():
            state = restore_training_state(recovery, model, optimizer, config)
            start_epoch = state["epoch"] + 1
            best, stale = state["best"], state["stale"]
            elapsed_before = state["elapsed"]
            torch.save(state["best_model"], checkpoint)
            print(json.dumps({"resumed_epoch": state["epoch"]}), flush=True)
        for epoch in range(start_epoch, args.epochs + 1):
            if stale >= args.patience:
                break
            model.train()
            for x, y in loaders["train"]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                if args.model.endswith("_noaugmentation"):
                    observed = torch.ones_like(x[..., :1])
                elif args.model.endswith("_independentaugmentation"):
                    rate = float(torch.empty(1).uniform_(0.0, 0.4))
                    observed = (torch.rand_like(x[..., :1]) >= rate).to(x.dtype)
                elif args.model.startswith("shiftguard"):
                    observed = structured_observation_mask(x)
                elif OUTAGE_AUGMENTED_SUFFIX in args.model:
                    # Baselines ignore `observed`, so zero the masked inputs exactly
                    # as evaluate_stress.infer does at test time.
                    observed = structured_observation_mask(x)
                    x = mask_inputs(x, observed)
                else:
                    observed = torch.ones_like(x[..., :1])
                backward_batch(model, x, y, observed, use_amp, args.microbatch_size)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            val_prediction, val_target = predict(model, loaders["val"], device, use_amp)
            val_mae = float(np.nanmean(np.abs(val_prediction - val_target)))
            print(json.dumps({"epoch": epoch, "val_mae": val_mae}), flush=True)
            if val_mae < best - 1e-5:
                best, stale = val_mae, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
            save_training_state(
                recovery, model, optimizer, config=config, epoch=epoch, best=best,
                stale=stale, elapsed=elapsed_before + time.perf_counter() - started,
                best_model=torch.load(checkpoint, map_location="cpu", weights_only=True),
            )
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    del loaders, train_x, train_y, val_x, val_y
    test_x, test_y = load_split(root, "test")
    test_loader = DataLoader(TensorDataset(test_x, test_y), inference_batch_size)
    prediction, target = predict(model, test_loader, device, use_amp)
    with np.load(root / "scaler.npz") as archive:
        mean, std = archive["mean"], archive["std"]
    prediction = prediction[..., 0] * std + mean
    target = target[..., 0] * std + mean
    results = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "amp": use_amp,
        "microbatch_size": args.microbatch_size,
        "parameters": trainable,
        "training_seconds": elapsed_before + time.perf_counter() - started,
        "metrics": horizon_metrics(target, prediction),
    }
    np.savez_compressed(output_dir / "predictions.npz", prediction=prediction, target=target)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
