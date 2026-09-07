from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from shiftguard.metrics import horizon_metrics


def sampled_features(
    x: np.ndarray, y: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = x.shape[2]
    times, sensors = rows // nodes, rows % nodes
    features = np.concatenate(
        [
            x[times, :, sensors, 0],
            x[times, :, sensors, 1],
            x[times, -1, sensors, 2:4],
        ],
        axis=1,
    )
    targets = y[times, :, sensors, 0]
    valid = np.all(y[times, :, sensors, 1] > 0.5, axis=1)
    return features[valid], targets[valid], valid


def dense_features(x: np.ndarray) -> np.ndarray:
    history = x[..., 0].transpose(0, 2, 1).reshape(-1, x.shape[1])
    masks = x[..., 1].transpose(0, 2, 1).reshape(-1, x.shape[1])
    calendar = x[:, -1, :, 2:4].reshape(-1, 2)
    return np.concatenate([history, masks, calendar], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Node-shared direct ridge baseline.")
    parser.add_argument("dataset", choices=["METR-LA", "PEMS-BAY"])
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-train-rows", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    root = args.processed_root / args.dataset
    with np.load(root / "train.npz") as split:
        train_x, train_y = split["x"], split["y"]
    total_rows = train_x.shape[0] * train_x.shape[2]
    count = min(args.max_train_rows, total_rows)
    rng = np.random.default_rng(args.seed)
    rows = rng.choice(total_rows, size=count, replace=False)
    features, targets, _ = sampled_features(train_x, train_y, rows)
    fit_rows = len(features)

    started = time.perf_counter()
    model = Ridge(alpha=args.alpha, solver="cholesky")
    model.fit(features, targets)
    training_seconds = time.perf_counter() - started
    del train_x, train_y, features, targets

    with np.load(root / "test.npz") as split:
        test_x, test_y = split["x"], split["y"]
    with np.load(root / "scaler.npz") as scaler:
        mean, std = scaler["mean"].ravel(), scaler["std"].ravel()
    predicted = model.predict(dense_features(test_x))
    predicted = predicted.reshape(test_x.shape[0], test_x.shape[2], -1).transpose(0, 2, 1)
    prediction = predicted * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    target = test_y[..., 0] * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    target[test_y[..., 1] < 0.5] = np.nan

    result = {
        "dataset": args.dataset,
        "model": "ridge_ar",
        "seed": args.seed,
        "metrics": horizon_metrics(target, prediction),
        "parameters": int(model.coef_.size + model.intercept_.size),
        "training_seconds": training_seconds,
        "sampled_rows": int(count),
        "complete_target_rows": fit_rows,
        "alpha": args.alpha,
    }
    destination = args.output_root / args.dataset / "ridge_ar" / f"seed_{args.seed}"
    destination.mkdir(parents=True, exist_ok=True)
    np.savez(destination / "model.npz", coef=model.coef_, intercept=model.intercept_)
    np.savez_compressed(destination / "predictions.npz", prediction=prediction, target=target)
    (destination / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
