"""Physically consistent sensor-outage timelines for deployment evaluation.

Unlike the window-level masks in :mod:`shiftguard.stress`, an outage here is a
property of a sensor at a timestamp. Consecutive forecast origins therefore see
the same failure, and a failed sensor's future readings are unobservable too.
"""

from __future__ import annotations

import numpy as np

from shiftguard.stress import _region

HISTORY = 12
HORIZON = 12


def episode_durations(native_observed: np.ndarray) -> np.ndarray:
    """Lengths of contiguous missing runs in a ``(T, N)`` availability trace."""

    missing = ~np.asarray(native_observed, dtype=bool)
    padded = np.pad(missing.astype(np.int8), ((1, 1), (0, 0)))
    change = np.diff(padded, axis=0)
    durations = []
    for node in range(missing.shape[1]):
        durations.append(np.flatnonzero(change[:, node] == -1) - np.flatnonzero(change[:, node] == 1))
    return np.concatenate(durations)


def sensor_renewal_timeline(
    steps: int, nodes: int, severity: float, durations: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Independent alternating-renewal failures with empirical down durations."""

    observed = np.ones((steps, nodes), dtype=bool)
    mean_up = float(durations.mean()) * (1 - severity) / severity
    for node in range(nodes):
        cursor = int(rng.integers(0, max(2, int(mean_up))))
        while cursor < steps:
            down = int(rng.choice(durations))
            observed[cursor : cursor + down, node] = False
            cursor += down + 1 + int(rng.exponential(mean_up))
    return observed


def _group_episodes(
    steps: int,
    nodes: int,
    severity: float,
    durations: np.ndarray,
    rng: np.random.Generator,
    draw_group,
) -> np.ndarray:
    observed = np.ones((steps, nodes), dtype=bool)
    long_durations = durations[durations >= HISTORY]
    while 1 - observed.mean() < severity:
        start = int(rng.integers(0, steps))
        down = int(rng.choice(long_durations))
        observed[start : start + down, draw_group()] = False
    return observed


def region_episode_timeline(
    steps: int,
    severity: float,
    durations: np.ndarray,
    adjacency: np.ndarray,
    rng: np.random.Generator,
    group_fraction: float = 0.15,
) -> np.ndarray:
    """Correlated-cluster failures: a new cluster fails in each episode."""

    nodes = len(adjacency)
    count = max(1, round(group_fraction * nodes))
    return _group_episodes(
        steps,
        nodes,
        severity,
        durations,
        rng,
        lambda: _region(adjacency, int(rng.integers(nodes)), count),
    )


def hub_episode_timeline(
    steps: int,
    severity: float,
    durations: np.ndarray,
    adjacency: np.ndarray,
    rng: np.random.Generator,
    pool_fraction: float = 0.30,
    group_fraction: float = 0.15,
) -> np.ndarray:
    """Episodes that remove a random subset of the highest-degree sensors."""

    nodes = len(adjacency)
    degree = np.count_nonzero(adjacency, axis=0) + np.count_nonzero(adjacency, axis=1)
    pool = np.argsort(degree)[-max(1, round(pool_fraction * nodes)) :]
    if severity >= len(pool) / nodes:
        raise ValueError("severity must be below the hub pool's share of sensors")
    count = max(1, round(group_fraction * nodes))
    return _group_episodes(
        steps, nodes, severity, durations, rng, lambda: rng.choice(pool, size=count, replace=False)
    )


def trace_replay_timeline(
    steps: int,
    nodes: int,
    trace: np.ndarray,
    copies: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replay real availability from a training period, keeping co-failures.

    ``copies`` superposes independent time offsets to raise severity. A trace
    from a network of another size is transplanted by sampling its columns.
    """

    observed = np.ones((steps, nodes), dtype=bool)
    length = len(trace)
    if length < steps:
        raise ValueError("trace is shorter than the replay timeline")
    for _ in range(copies):
        offset = int(rng.integers(0, length - steps + 1))
        columns = (
            np.arange(nodes)
            if trace.shape[1] == nodes
            else rng.choice(trace.shape[1], size=nodes, replace=True)
        )
        observed &= trace[offset : offset + steps][:, columns]
    return observed


def windows_from_timeline(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(S,P,N)`` input and ``(S,H,N)`` target availability per origin."""

    view = np.lib.stride_tricks.sliding_window_view(observed, HISTORY + HORIZON, axis=0)
    view = view.transpose(0, 2, 1)
    return view[:, :HISTORY], view[:, HISTORY:]


def historical_profile(
    speed: np.ndarray, available: np.ndarray, time_index: np.ndarray, day_index: np.ndarray
) -> np.ndarray:
    """Training-only time-of-week mean per sensor, shape ``(7, 288, N)``."""

    nodes = speed.shape[1]
    totals = np.zeros((7, 288, nodes))
    counts = np.zeros((7, 288, nodes))
    valid = available.astype(bool)
    np.add.at(totals, (day_index, time_index), np.where(valid, speed, 0.0))
    np.add.at(counts, (day_index, time_index), valid.astype(float))
    pooled = np.where(valid, speed, 0.0).sum(0) / np.maximum(valid.sum(0), 1)
    return np.where(counts > 0, totals / np.maximum(counts, 1), pooled[None, None])
