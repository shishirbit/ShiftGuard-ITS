from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from train import (
    MODEL_CHOICES,
    base_model_name,
    build_model,
    correlation_adjacency,
    load_split,
    unpack,
)

from shiftguard.calibration import (
    finite_mean,
    interval_metrics,
    prequential_calibration,
    risk_coverage_curve,
)
from shiftguard.metrics import horizon_metrics
from shiftguard.stress import burst_mask, hub_mask, independent_mask, region_mask


def stress_masks(
    artifact_root: Path,
    dataset: str,
    seed: int,
    rate: float,
    shape: tuple[int, ...],
    adjacency: np.ndarray,
) -> dict[str, np.ndarray]:
    """Create or load model-independent masks for paired comparisons."""

    destination = artifact_root / "stress_masks" / dataset / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"rate_{rate:.1f}.npz"
    if path.exists():
        with np.load(path) as archive:
            masks = {name: archive[name] for name in archive.files}
        if all(mask.shape == shape for mask in masks.values()):
            return {"clean": np.ones(shape, dtype=bool), **masks}
        raise ValueError(f"saved stress masks have incompatible shape: {path}")
    rng = np.random.default_rng(seed)
    masks = {
        "independent": independent_mask(shape, rate, rng),
        "burst": burst_mask(shape, rate, rng),
        "region": region_mask(shape, rate, adjacency, rng),
        "hub": hub_mask(shape, rate, adjacency),
    }
    np.savez_compressed(path, **masks)
    return {"clean": np.ones(shape, dtype=bool), **masks}


@torch.no_grad()
def infer(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    observed: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = TensorDataset(x, y, torch.from_numpy(observed.astype(np.float32)))
    points, scales, targets = [], [], []
    model.eval()
    for bx, by, bm in DataLoader(dataset, batch_size=batch_size):
        bx, bm = bx.to(device), bm.to(device)
        bx = bx.clone()
        bx[..., :1] = bx[..., :1] * bm
        point, scale = unpack(model(bx, bm))
        points.append(point.cpu().numpy()[..., 0])
        scales.append(
            scale.cpu().numpy()[..., 0] if scale is not None else np.ones_like(points[-1])
        )
        target = by.numpy()[..., 0]
        target[by.numpy()[..., 1] == 0] = np.nan
        targets.append(target)
    return np.concatenate(points), np.concatenate(scales), np.concatenate(targets)


def regime_labels(raw_x: np.ndarray, raw_y: np.ndarray) -> np.ndarray:
    last = raw_x[:, -1, :, 0].mean(axis=1)
    future = np.nanmean(raw_y[..., 0], axis=(1, 2))
    labels = np.full(len(last), "stable_mid", dtype="U16")
    labels[(last >= 55) & (future >= 55)] = "free"
    labels[(last <= 25) & (future <= 25)] = "congested"
    labels[((last >= 55) & (future < 55)) | ((last <= 25) & (future > 25))] = "transition"
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["METR-LA", "PEMS-BAY"], required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--calibration",
        choices=["group-prequential", "global-prequential", "group-static"],
        default="group-prequential",
    )
    parser.add_argument("--selection", choices=["risk", "all"], default="risk")
    args = parser.parse_args()
    if base_model_name(args.model) == "staeformer":
        args.batch_size = min(args.batch_size, 24)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = args.data_root / args.dataset
    val_x, val_y = load_split(root, "val")
    test_x, test_y = load_split(root, "test")
    adjacency_path = root / "correlation_adjacency.npy"
    if adjacency_path.exists():
        adjacency = np.load(adjacency_path)
    else:
        train_x, _ = load_split(root, "train")
        adjacency = correlation_adjacency(train_x).numpy()
        temporary = adjacency_path.with_suffix(".tmp.npy")
        np.save(temporary, adjacency)
        temporary.replace(adjacency_path)
    model = build_model(args.model, test_x.shape[2], torch.from_numpy(adjacency)).to(device)
    checkpoint = args.artifact_root / args.dataset / args.model / f"seed_{args.seed}" / "best.pt"
    if sum(p.numel() for p in model.parameters() if p.requires_grad):
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    with np.load(root / "scaler.npz") as archive:
        mean, std = archive["mean"], archive["std"]
    inverse = lambda array: array * std + mean

    mask_shape = (*val_x.shape[:-1], 1)
    clean_val = np.ones(mask_shape, dtype=bool)
    vp, vs, vt = infer(model, val_x, val_y, clean_val, device, args.batch_size)
    vp, vt, vs = inverse(vp), inverse(vt), vs * std
    mask_shape = (*test_x.shape[:-1], 1)
    families = stress_masks(
        args.artifact_root, args.dataset, args.seed, args.rate, mask_shape, adjacency
    )
    raw_x, raw_y = inverse(test_x.numpy()[..., 0])[..., None], inverse(test_y.numpy()[..., 0])[..., None]
    regimes = regime_labels(raw_x, raw_y)
    results: dict[str, object] = {}
    for family, observed in families.items():
        point, scale, target = infer(model, test_x, test_y, observed, device, args.batch_size)
        point, target, scale = inverse(point), inverse(target), scale * std
        lower, upper, accepted, rules = prequential_calibration(
            vt,
            vp,
            vs,
            target,
            point,
            scale,
            alpha=0.1,
            groupwise=args.calibration != "global-prequential",
            update=args.calibration != "group-static",
            risk_control=args.selection == "risk",
        )
        width = upper - lower
        origin_error = finite_mean(np.abs(target - point), (1, 2))
        origin_uncertainty = finite_mean(width, (1, 2))
        _, _, aurc = risk_coverage_curve(origin_error, origin_uncertainty)
        accepted_errors = np.abs(target - point)[accepted]
        accepted_errors = accepted_errors[np.isfinite(accepted_errors)]
        accepted_origins = accepted[:, 0, 0]
        accepted_origin_errors = origin_error[accepted_origins]
        accepted_origin_errors = accepted_origin_errors[np.isfinite(accepted_origin_errors)]
        finite_interval = np.isfinite(target) & np.isfinite(lower) & np.isfinite(upper)
        covered = finite_interval & (target >= lower) & (target <= upper)
        group_count = finite_interval.sum(axis=0)
        node_horizon_coverage = np.divide(
            covered.sum(axis=0),
            group_count,
            out=np.full(group_count.shape, np.nan, dtype=np.float64),
            where=group_count > 0,
        )
        regime_coverage = {}
        for regime in np.unique(regimes):
            selected = regimes == regime
            valid = finite_interval[selected]
            valid_count = int(valid.sum())
            regime_coverage[str(regime)] = (
                float(covered[selected].sum() / valid_count)
                if valid_count
                else float("nan")
            )
        record: dict[str, object] = {
            "point": horizon_metrics(target, point),
            "interval": interval_metrics(target, lower, upper, alpha=0.1),
            "interval_groups": {
                "worst_node_horizon_coverage": float(np.nanmin(node_horizon_coverage)),
                "worst_regime_coverage": float(
                    np.nanmin(list(regime_coverage.values()))
                ),
                "regime_coverage": regime_coverage,
            },
            "aurc": aurc,
            "selective": {
                "coverage": float(np.mean(accepted)),
                "mae": float(np.mean(accepted_errors)) if len(accepted_errors) else None,
                "failure_rate_5mph": (
                    float(np.mean(accepted_origin_errors > 5.0))
                    if len(accepted_origin_errors)
                    else None
                ),
                "prequential_blocks": [rule.__dict__ for rule in rules],
            },
            "regimes": {},
        }
        for regime in np.unique(regimes):
            selected = regimes == regime
            record["regimes"][str(regime)] = horizon_metrics(target[selected], point[selected])["average"]
        results[family] = record
    destination = args.artifact_root / args.dataset / args.model / f"seed_{args.seed}"
    destination.mkdir(parents=True, exist_ok=True)
    suffix = "" if (args.calibration, args.selection) == ("group-prequential", "risk") else (
        f"_{args.calibration}_{args.selection}"
    )
    payload = {
        "protocol_version": "origin-risk-v1",
        "group_coverage_version": "node-horizon-regime-v1",
        "calibration": args.calibration,
        "selection": args.selection,
        "risk_unit": "forecast_origin_mean_absolute_error",
        "stress_rate": args.rate,
        "results": results,
    }
    (destination / f"stress_rate_{args.rate:.1f}{suffix}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
