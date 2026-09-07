from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from shiftguard.metrics import horizon_metrics


def inverse(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return values * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def future_week_slots(x: np.ndarray, horizon: int) -> np.ndarray:
    time_of_day = np.rint(x[:, -1, 0, 2] * 288).astype(int).clip(0, 287)
    day_of_week = np.rint(x[:, -1, 0, 3] * 6).astype(int).clip(0, 6)
    origin = day_of_week * 288 + time_of_day
    return (origin[:, None] + np.arange(1, horizon + 1)[None]) % (7 * 288)


def historical_average(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    """Training-only sensor-by-time-of-week mean, evaluated at each horizon."""

    nodes = train_x.shape[2]
    slots = future_week_slots(train_x, 1)[:, 0]
    values, observed = train_y[:, 0, :, 0], train_y[:, 0, :, 1] > 0.5
    sums = np.zeros((7 * 288, nodes), dtype=np.float64)
    counts = np.zeros_like(sums)
    node_index = np.arange(nodes)[None, :]
    np.add.at(sums, (slots[:, None], node_index), np.where(observed, values, 0.0))
    np.add.at(counts, (slots[:, None], node_index), observed)
    sensor_mean = np.divide(
        np.where(observed, values, 0.0).sum(0),
        observed.sum(0),
        out=np.zeros(nodes),
        where=observed.sum(0) > 0,
    )
    profile = np.divide(
        sums,
        counts,
        out=np.broadcast_to(sensor_mean, sums.shape).copy(),
        where=counts > 0,
    )
    return profile[future_week_slots(test_x, 12)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate training-free baselines.")
    parser.add_argument("dataset", choices=["METR-LA", "PEMS-BAY"])
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    started = time.perf_counter()
    root = args.processed_root / args.dataset
    with np.load(root / "train.npz") as split:
        train_x, train_y = split["x"], split["y"]
    with np.load(root / "test.npz") as split:
        x, y = split["x"], split["y"]
    with np.load(root / "scaler.npz") as scaler:
        mean, std = scaler["mean"].ravel(), scaler["std"].ravel()

    target = inverse(y[..., 0], mean, std)
    target[y[..., 1] < 0.5] = np.nan
    prediction = inverse(
        np.repeat(x[:, -1:, :, 0], y.shape[1], axis=1), mean, std
    )
    persistence_result = {
        "dataset": args.dataset,
        "model": "persistence",
        "seed": 0,
        "metrics": horizon_metrics(target, prediction),
        "parameters": 0,
        "training_seconds": 0.0,
        "evaluation_seconds": time.perf_counter() - started,
        "protocol": "chronological 70/10/20 split; observed targets only",
    }
    ha_prediction = inverse(historical_average(train_x, train_y, x), mean, std)
    ha_result = {
        "dataset": args.dataset,
        "model": "historical_average",
        "seed": 0,
        "metrics": horizon_metrics(target, ha_prediction),
        "parameters": 0,
        "training_seconds": 0.0,
        "evaluation_seconds": time.perf_counter() - started,
        "protocol": "training-only sensor-by-time-of-week mean",
    }
    for model, result, model_prediction in (
        ("persistence", persistence_result, prediction),
        ("historical_average", ha_result, ha_prediction),
    ):
        destination = args.output_root / args.dataset / model / "seed_0"
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / "predictions.npz", prediction=model_prediction, target=target
        )
        (destination / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    print(json.dumps({"persistence": persistence_result, "historical_average": ha_result}, indent=2))


if __name__ == "__main__":
    main()
