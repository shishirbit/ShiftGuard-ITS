"""Evidence-gated forecasting: use a sensor's data freshness at inference.

A forecaster's output for a sensor that has not reported cannot rest on that
sensor's recent history. These helpers blend such forecasts with a historical
profile and calibrate intervals per freshness level. Sensors that are working
are hidden on purpose ("shadow blinding") to obtain residuals for silent
sensors, whose own outcomes never arrive in deployment.
"""

from __future__ import annotations

import numpy as np

from shiftguard.calibration import conformal_quantile

FRESH, RECENT, STALE, BLIND = 0, 1, 2, 3
BUCKET_NAMES = ("fresh", "recent", "stale", "blind")
MIN_POOL = 50


def evidence_buckets(input_up: np.ndarray) -> np.ndarray:
    """Map ``(S,P,N)`` availability to a ``(S,N)`` freshness bucket.

    Fresh: reported at the origin. Recent: last report 1--5 steps ago. Stale:
    6 or more steps ago. Blind: no report anywhere in the input window.
    """

    history = input_up.shape[1]
    age = input_up[:, ::-1].argmax(axis=1)
    bucket = np.full(age.shape, STALE, dtype=np.int8)
    bucket[age == 0] = FRESH
    bucket[(age >= 1) & (age <= 5)] = RECENT
    bucket[~input_up.any(axis=1)] = BLIND
    assert history > 5
    return bucket


def shadow_availability(
    input_up: np.ndarray, rng: np.random.Generator, blind: float = 0.15, suffix: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Hide some reporting sensors; return new availability and hidden sensors."""

    origins, history, nodes = input_up.shape
    draw = rng.random((origins, nodes))
    hide_all = draw < blind
    hide_suffix = (draw >= blind) & (draw < blind + suffix)
    start = rng.integers(1, history, size=(origins, nodes))
    steps = np.arange(history)[None, :, None]
    hidden_cells = hide_all[:, None, :] | (hide_suffix[:, None, :] & (steps >= start[:, None, :]))
    return input_up & ~hidden_cells, hide_all | hide_suffix


def fit_blend_weights(
    target: np.ndarray,
    point: np.ndarray,
    profile: np.ndarray,
    bucket: np.ndarray,
    grid: int = 21,
) -> np.ndarray:
    """Per bucket and horizon, the model weight minimizing validation MAE."""

    horizon = target.shape[1]
    weights = np.ones((len(BUCKET_NAMES), horizon))
    candidates = np.linspace(0.0, 1.0, grid)
    for level in range(len(BUCKET_NAMES)):
        selected = bucket == level
        if not selected.any():
            continue
        for step in range(horizon):
            y, p, f = target[:, step][selected], point[:, step][selected], profile[:, step][selected]
            valid = np.isfinite(y)
            if valid.sum() < MIN_POOL:
                continue
            y, p, f = y[valid], p[valid], f[valid]
            losses = [np.abs(y - (w * p + (1 - w) * f)).mean() for w in candidates]
            weights[level, step] = candidates[int(np.argmin(losses))]
    return weights


def blend(
    point: np.ndarray, profile: np.ndarray, bucket: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Apply bucket-and-horizon weights to ``(S,H,N)`` forecasts."""

    horizon = point.shape[1]
    w = weights[bucket[:, None, :], np.arange(horizon)[None, :, None]]
    return w * point + (1 - w) * profile


def bucket_quantiles(
    score: np.ndarray, bucket: np.ndarray, alpha: float, shadow: tuple[np.ndarray, np.ndarray] | None = None
) -> np.ndarray:
    """Conformal quantile per ``(bucket, horizon)``, pooling thin buckets."""

    horizon = score.shape[1]
    table = np.empty((len(BUCKET_NAMES), horizon))
    for step in range(horizon):
        pools = []
        for level in range(len(BUCKET_NAMES)):
            values = [score[:, step][bucket == level]]
            if shadow is not None:
                values.append(shadow[0][:, step][shadow[1] == level])
            pool = np.concatenate(values)
            pools.append(pool[np.isfinite(pool)])
        everything = np.concatenate(pools)
        for level, pool in enumerate(pools):
            table[level, step] = conformal_quantile(
                pool if len(pool) >= MIN_POOL else everything, alpha
            )
    return table


def evidence_intervals(
    prior: dict[str, np.ndarray],
    live: dict[str, np.ndarray],
    point: np.ndarray,
    scale: np.ndarray,
    *,
    alpha: float = 0.1,
    block_size: int = 288,
    window: int = 2016,
) -> tuple[np.ndarray, np.ndarray]:
    """Block-prequential intervals with freshness-specific conformal quantiles.

    ``prior`` and ``live`` hold normalized residuals and buckets for real and
    shadow forecasts. Residuals must be NaN wherever the outcome was unobserved.
    Only origins whose whole horizon has elapsed enter the calibration window.
    """

    horizon = point.shape[1]
    lower = np.empty_like(point)
    upper = np.empty_like(point)
    keys = ("score", "bucket", "shadow_score", "shadow_bucket")
    for start in range(0, len(point), block_size):
        stop = min(start + block_size, len(point))
        known = max(0, start - horizon)
        history = {
            key: np.concatenate([prior[key], live[key][:known]])[-window:] for key in keys
        }
        table = bucket_quantiles(
            history["score"], history["bucket"], alpha,
            (history["shadow_score"], history["shadow_bucket"]),
        )
        level = live["bucket"][start:stop]
        quantile = table[level[:, None, :], np.arange(horizon)[None, :, None]]
        radius = quantile * scale[start:stop]
        lower[start:stop] = point[start:stop] - radius
        upper[start:stop] = point[start:stop] + radius
    return lower, upper
