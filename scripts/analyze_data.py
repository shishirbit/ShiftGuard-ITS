from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from shiftguard.data import load_csv


def safe_corr(values: np.ndarray, lag: int) -> float:
    left, right = values[:-lag].ravel(), values[lag:].ravel()
    valid = np.isfinite(left) & np.isfinite(right) & (left > 0) & (right > 0)
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["METR-LA", "PEMS-BAY"])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eda"))
    args = parser.parse_args()
    frame = load_csv(args.raw_dir / f"{args.dataset}.csv")
    values = frame.to_numpy(dtype=np.float64)
    observed = np.isfinite(values) & (values > 0)
    clean = np.where(observed, values, np.nan)
    sensor_transition = np.abs(np.diff(clean, axis=0)) >= 5
    summary = {
        "dataset": args.dataset,
        "timestamps": len(frame),
        "sensors": frame.shape[1],
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "missing_fraction": float(1 - observed.mean()),
        "mean_speed": float(np.nanmean(clean)),
        "std_speed": float(np.nanstd(clean)),
        "congested_fraction_below_25": float(np.nanmean(clean < 25)),
        "free_fraction_above_55": float(np.nanmean(clean > 55)),
        "sensor_transition_fraction_5mph": float(np.nanmean(sensor_transition)),
        "autocorrelation": {
            "lag_1_5min": safe_corr(clean, 1),
            "lag_12_1hour": safe_corr(clean, 12),
            "lag_288_1day": safe_corr(clean, 288),
            "lag_2016_1week": safe_corr(clean, 2016),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.dataset}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    sensor = pd.DataFrame(
        {
            "sensor_id": frame.columns,
            "mean": np.nanmean(clean, axis=0),
            "std": np.nanstd(clean, axis=0),
            "missing_fraction": 1 - observed.mean(axis=0),
            "congested_fraction": np.nanmean(clean < 25, axis=0),
        }
    )
    sensor.to_csv(args.output_dir / f"{args.dataset}_sensors.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
