from __future__ import annotations

import numpy as np


def point_metrics(target: np.ndarray, prediction: np.ndarray, threshold: float = 0.0) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(prediction) & (np.abs(target) > threshold)
    error = prediction[valid] - target[valid]
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(np.abs(error / target[valid])) * 100),
    }


def horizon_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, dict[str, float]]:
    result = {"average": point_metrics(target, prediction)}
    for horizon in (3, 6, 12):
        if horizon <= target.shape[1]:
            result[f"h{horizon}"] = point_metrics(target[:, horizon - 1], prediction[:, horizon - 1])
    return result

