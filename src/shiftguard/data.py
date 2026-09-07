"""Leakage-controlled data preparation for METR-LA and PEMS-BAY."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    x: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


def load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"duplicate timestamps in {path}")
    return frame.astype(np.float32)


def window(
    input_values: np.ndarray,
    target_values: np.ndarray,
    observed: np.ndarray,
    timestamps: np.ndarray,
    history: int,
    horizon: int,
) -> Split:
    count = len(input_values) - history - horizon + 1
    if count <= 0:
        raise ValueError("series is shorter than history + horizon")
    x = np.empty((count, history, input_values.shape[1], 4), dtype=np.float32)
    y = np.empty((count, horizon, input_values.shape[1], 2), dtype=np.float32)
    target_times = np.empty((count, horizon), dtype="datetime64[ns]")
    date_index = pd.DatetimeIndex(timestamps)
    time_of_day = np.asarray(
        (date_index.hour * 3600 + date_index.minute * 60 + date_index.second) / 86400.0,
        dtype=np.float32,
    )
    day_of_week = np.asarray(date_index.dayofweek / 6.0, dtype=np.float32)
    for i in range(count):
        x[i, ..., 0] = input_values[i : i + history]
        x[i, ..., 1] = observed[i : i + history]
        x[i, ..., 2] = time_of_day[i : i + history, None]
        x[i, ..., 3] = day_of_week[i : i + history, None]
        y[i, ..., 0] = target_values[i + history : i + history + horizon]
        y[i, ..., 1] = observed[i + history : i + history + horizon]
        target_times[i] = timestamps[i + history : i + history + horizon]
    return Split(x=x, y=y, timestamps=target_times)


def prepare(
    frame: pd.DataFrame,
    history: int = 12,
    horizon: int = 12,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> tuple[dict[str, Split], Standardizer]:
    """Split raw time first, fit normalization on train only, then window.

    Windows may use the preceding ``history`` observations at validation/test
    boundaries, but no target or normalization statistic crosses backward.
    """

    n = len(frame)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    raw = frame.to_numpy(dtype=np.float32)
    observed = np.isfinite(raw) & (raw > 0)
    train_raw = np.where(observed[:train_end], raw[:train_end], np.nan)
    mean = np.nanmean(train_raw, axis=0, keepdims=True).astype(np.float32)
    std = np.nanstd(train_raw, axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    scaler = Standardizer(mean=mean, std=std)
    normalized_input = scaler.transform(np.where(observed, raw, mean))
    normalized_target = scaler.transform(np.where(observed, raw, 0.0))
    times = frame.index.to_numpy(dtype="datetime64[ns]")

    boundaries = {
        "train": (0, train_end),
        "val": (train_end - history, val_end),
        "test": (val_end - history, n),
    }
    splits: dict[str, Split] = {}
    for name, (start, stop) in boundaries.items():
        candidate = window(
            normalized_input[start:stop],
            normalized_target[start:stop],
            observed[start:stop],
            times[start:stop],
            history,
            horizon,
        )
        if name != "train":
            lower = times[train_end] if name == "val" else times[val_end]
            keep = candidate.timestamps[:, 0] >= lower
            candidate = Split(candidate.x[keep], candidate.y[keep], candidate.timestamps[keep])
        splits[name] = candidate
    return splits, scaler


def save_processed(
    csv_path: Path,
    output_dir: Path,
    history: int = 12,
    horizon: int = 12,
) -> None:
    frame = load_csv(csv_path)
    splits, scaler = prepare(frame, history=history, horizon=horizon)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        np.savez_compressed(
            output_dir / f"{name}.npz",
            x=split.x,
            y=split.y,
            timestamps=split.timestamps,
        )
    np.savez_compressed(
        output_dir / "scaler.npz",
        mean=scaler.mean,
        std=scaler.std,
        sensor_ids=np.asarray(frame.columns),
    )
