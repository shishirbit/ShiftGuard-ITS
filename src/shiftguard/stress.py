"""Deterministic, auditable sensor-outage stress generators."""

from __future__ import annotations

import numpy as np


def _check_shape(shape: tuple[int, int, int, int]) -> None:
    if len(shape) != 4 or shape[-1] != 1:
        raise ValueError("expected (samples, history, nodes, 1)")


def independent_mask(
    shape: tuple[int, int, int, int], rate: float, rng: np.random.Generator
) -> np.ndarray:
    _check_shape(shape)
    return rng.random(shape) >= rate


def burst_mask(
    shape: tuple[int, int, int, int], rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Mask complete temporal suffixes for a sampled subset of sensor/windows."""

    _check_shape(shape)
    samples, history, nodes, _ = shape
    observed = np.ones(shape, dtype=bool)
    sensor_windows = max(1, round(rate * samples * nodes))
    chosen = rng.choice(samples * nodes, size=sensor_windows, replace=False)
    for flat in chosen:
        sample, node = divmod(int(flat), nodes)
        start = int(rng.integers(0, history))
        observed[sample, start:, node, 0] = False
    return observed


def _region(adjacency: np.ndarray, seed: int, count: int) -> np.ndarray:
    """Grow a contiguous region using strongest unseen neighbors."""

    selected = [seed]
    unseen = set(range(len(adjacency))) - {seed}
    while len(selected) < count and unseen:
        candidate = max(
            unseen,
            key=lambda node: float(np.max(adjacency[node, selected])),
        )
        selected.append(candidate)
        unseen.remove(candidate)
    return np.asarray(selected, dtype=np.int64)


def region_mask(
    shape: tuple[int, int, int, int],
    rate: float,
    adjacency: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    _check_shape(shape)
    if adjacency.shape != (shape[2], shape[2]):
        raise ValueError("adjacency and node count disagree")
    count = max(1, round(rate * shape[2]))
    failed = _region(adjacency, int(rng.integers(shape[2])), count)
    observed = np.ones(shape, dtype=bool)
    observed[:, :, failed, :] = False
    return observed


def hub_mask(shape: tuple[int, int, int, int], rate: float, adjacency: np.ndarray) -> np.ndarray:
    _check_shape(shape)
    count = max(1, round(rate * shape[2]))
    degree = np.count_nonzero(adjacency, axis=0) + np.count_nonzero(adjacency, axis=1)
    failed = np.argsort(degree)[-count:]
    observed = np.ones(shape, dtype=bool)
    observed[:, :, failed, :] = False
    return observed


def apply_outage(x: np.ndarray, observed: np.ndarray, fill: float = 0.0) -> np.ndarray:
    if x.shape != observed.shape:
        raise ValueError("x and observed mask must have equal shapes")
    return np.where(observed, x, np.asarray(fill, dtype=x.dtype))

