from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from train import build_model, correlation_adjacency, load_split


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def timed_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    observed: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        model(x, observed)
    synchronize(device)
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        model(x, observed)
        synchronize(device)
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples), statistics.quantiles(samples, n=100)[94]


@torch.inference_mode()
def profiler_flops(model: torch.nn.Module, x: torch.Tensor, observed: torch.Tensor) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if x.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, with_flops=True) as profile:
        model(x, observed)
    return int(sum(event.flops or 0 for event in profile.key_averages()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["METR-LA", "PEMS-BAY"])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/summary/efficiency.csv"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        root = args.data_root / dataset
        test_x, _ = load_split(root, "test")
        adjacency_path = root / "correlation_adjacency.npy"
        if adjacency_path.exists():
            adjacency = torch.from_numpy(np.load(adjacency_path))
        else:
            train_x, _ = load_split(root, "train")
            adjacency = correlation_adjacency(train_x)
            np.save(adjacency_path, adjacency.numpy())
        for name in args.models:
            checkpoint = args.artifact_root / dataset / name / f"seed_{args.seed}" / "best.pt"
            model = build_model(name, test_x.shape[2], adjacency).to(device).eval()
            if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):
                if not checkpoint.exists():
                    print(f"missing {checkpoint}", file=sys.stderr)
                    continue
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            parameters = sum(parameter.numel() for parameter in model.parameters())
            for batch_size in args.batch_sizes:
                x = test_x[:batch_size].to(device)
                observed = torch.ones_like(x[..., :1])
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                median_ms, p95_ms = timed_forward(
                    model, x, observed, device, args.warmup, args.repeats
                )
                peak_memory_mb = (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == "cuda"
                    else 0.0
                )
                flops = profiler_flops(model, x[:1], observed[:1]) if batch_size == 1 else None
                row = {
                    "dataset": dataset,
                    "model": name,
                    "seed": args.seed,
                    "device": str(device),
                    "batch_size": batch_size,
                    "parameters": parameters,
                    "profiler_flops_per_sample": flops,
                    "median_latency_ms": median_ms,
                    "p95_latency_ms": p95_ms,
                    "throughput_samples_per_second": batch_size * 1000 / median_ms,
                    "peak_allocated_memory_mb": peak_memory_mb,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
