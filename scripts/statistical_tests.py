from __future__ import annotations

import argparse
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from shiftguard.registry import REFERENCE_MODEL


@cache
def origin_mae(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        prediction, target = archive["prediction"], archive["target"]
    error = np.abs(prediction - target)
    count = np.isfinite(error).sum(axis=(1, 2))
    return np.divide(
        np.nansum(error, axis=(1, 2)),
        count,
        out=np.full(len(error), np.nan),
        where=count > 0,
    )


def block_means(values: np.ndarray, block_size: int) -> np.ndarray:
    blocks = [values[start : start + block_size] for start in range(0, len(values), block_size)]
    return np.asarray([np.nanmean(block) for block in blocks if np.isfinite(block).any()])


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired daily-block inference for model MAE.")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--proposed", default=REFERENCE_MODEL)
    parser.add_argument("--block-size", type=int, default=288)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    deterministic = {"historical_average", "persistence", "ridge_ar"}
    rows: list[dict[str, object]] = []
    for dataset_dir in sorted(path for path in args.artifact_root.iterdir() if path.is_dir()):
        proposed_paths = sorted((dataset_dir / args.proposed).glob("seed_*/predictions.npz"))
        if not proposed_paths:
            continue
        for model_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            if model_dir.name == args.proposed:
                continue
            baseline_paths = sorted(model_dir.glob("seed_*/predictions.npz"))
            if not baseline_paths:
                continue
            if model_dir.name not in deterministic and len(baseline_paths) < len(proposed_paths):
                continue
            by_seed = {path.parent.name: path for path in baseline_paths}
            differences, proposed_errors, baseline_errors = [], [], []
            for proposed_path in proposed_paths:
                baseline_path = by_seed.get(proposed_path.parent.name, baseline_paths[0])
                proposed_mae, baseline_mae = origin_mae(proposed_path), origin_mae(baseline_path)
                if proposed_mae.shape != baseline_mae.shape:
                    raise ValueError(f"incompatible prediction arrays: {proposed_path}, {baseline_path}")
                differences.append(block_means(proposed_mae - baseline_mae, args.block_size))
                proposed_errors.append(np.nanmean(proposed_mae))
                baseline_errors.append(np.nanmean(baseline_mae))
            paired = np.stack(differences)
            seed_draws = rng.integers(0, paired.shape[0], size=(args.bootstrap, paired.shape[0]))
            block_draws = rng.integers(0, paired.shape[1], size=(args.bootstrap, paired.shape[1]))
            draws = paired[seed_draws[:, :, None], block_draws[:, None, :]].mean(axis=(1, 2))
            daily_differences = paired.mean(axis=0)
            statistic = wilcoxon(
                daily_differences,
                alternative="two-sided",
                zero_method="wilcox",
            )
            proposed_mean, baseline_mean = np.mean(proposed_errors), np.mean(baseline_errors)
            rows.append(
                {
                    "dataset": dataset_dir.name,
                    "baseline": model_dir.name,
                    "proposed_mae": proposed_mean,
                    "baseline_mae": baseline_mean,
                    "delta_mae": proposed_mean - baseline_mean,
                    "relative_improvement_pct": 100 * (baseline_mean - proposed_mean) / baseline_mean,
                    "ci95_low": np.quantile(draws, 0.025),
                    "ci95_high": np.quantile(draws, 0.975),
                    "wilcoxon_p": statistic.pvalue,
                    "daily_blocks": paired.shape[1],
                    "proposed_seeds": len(proposed_paths),
                }
            )
    if not rows:
        raise SystemExit("no paired prediction artifacts found")
    adjusted = holm_adjust([float(row["wilcoxon_p"]) for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_p"] = value
    frame = pd.DataFrame(rows)
    destination = args.artifact_root / "summary"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "paired_block_tests.csv", index=False)
    display = frame[~frame["baseline"].str.startswith("shiftguard_")][
        ["dataset", "baseline", "delta_mae", "ci95_low", "ci95_high", "relative_improvement_pct", "holm_p"]
    ].rename(
        columns={
            "dataset": "Dataset",
            "baseline": "Comparator",
            "delta_mae": "Delta MAE",
            "ci95_low": "CI low",
            "ci95_high": "CI high",
            "relative_improvement_pct": "Improvement (pct.)",
            "holm_p": "Holm p",
        }
    )
    display["Comparator"] = display["Comparator"].str.replace("_", " ")
    display["Holm p"] = display["Holm p"].map(
        lambda value: r"$<0.0001$" if value < 0.0001 else f"{value:.4f}"
    )
    latex = display.to_latex(index=False, float_format="%.4f", escape=False)
    (destination / "paired_block_tests.tex").write_text(latex, encoding="utf-8")
    paper = Path("paper")
    if paper.is_dir():
        (paper / "generated_statistics.tex").write_text(latex, encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
