from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PRIMARY_MODELS = (
    "persistence",
    "linear",
    "mlp",
    "gru",
    "stgcn",
    "dcrnn",
    "graphwavenet",
    "agcrn",
    "stid",
    "staeformer",
)
ABLATIONS = (
    "shiftguard_noreliability",
    "shiftguard_staticgraph",
    "shiftguard_adaptivegraph",
    "shiftguard_norouter",
    "shiftguard_noaugmentation",
    "shiftguard_independentaugmentation",
)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def compare(
    frame: pd.DataFrame,
    competitors: tuple[str, ...],
    metrics: tuple[str, ...],
    bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    proposed = frame[frame.model == "shiftguard"]
    for dataset in sorted(frame.dataset.unique()):
        proposed_dataset = proposed[proposed.dataset == dataset]
        for competitor in competitors:
            baseline = frame[(frame.dataset == dataset) & (frame.model == competitor)]
            for metric in metrics:
                paired = proposed_dataset.merge(
                    baseline,
                    on=["dataset", "seed", "rate", "family"],
                    suffixes=("_proposed", "_baseline"),
                    validate="one_to_one",
                )
                paired["difference"] = paired[f"{metric}_proposed"] - paired[f"{metric}_baseline"]
                matrix = paired.pivot(index="seed", columns=["rate", "family"], values="difference")
                if matrix.isna().any().any():
                    raise ValueError(f"incomplete pairing for {dataset}/{competitor}/{metric}")
                values = matrix.to_numpy()
                seed_draws = rng.integers(0, values.shape[0], size=(bootstrap, values.shape[0]))
                scenario_draws = rng.integers(
                    0, values.shape[1], size=(bootstrap, values.shape[1])
                )
                draws = values[seed_draws[:, :, None], scenario_draws[:, None, :]].mean((1, 2))
                seed_means = values.mean(axis=1)
                test = wilcoxon(seed_means, alternative="two-sided", zero_method="wilcox")
                rows.append(
                    {
                        "dataset": dataset,
                        "competitor": competitor,
                        "metric": metric,
                        "proposed_mean": paired[f"{metric}_proposed"].mean(),
                        "competitor_mean": paired[f"{metric}_baseline"].mean(),
                        "difference": paired.difference.mean(),
                        "ci95_low": np.quantile(draws, 0.025),
                        "ci95_high": np.quantile(draws, 0.975),
                        "wilcoxon_p": test.pvalue,
                        "seeds": values.shape[0],
                        "scenarios": values.shape[1],
                    }
                )
    result = pd.DataFrame(rows)
    result["holm_p"] = np.nan
    for _, indices in result.groupby(["dataset", "metric"]).groups.items():
        result.loc[indices, "holm_p"] = holm_adjust(result.loc[indices, "wilcoxon_p"].tolist())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchical paired inference for stress results.")
    parser.add_argument("--input", type=Path, default=Path("artifacts/summary/stress_all_runs.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/summary"))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input)
    primary = raw[
        raw.model.isin(("shiftguard", *PRIMARY_MODELS))
        & (raw.calibration == "group-prequential")
        & (raw.selection == "risk")
        & (raw.family != "clean")
        & raw.rate.isin((0.1, 0.2, 0.3, 0.4, 0.5))
    ]
    ablations = raw[
        raw.model.isin(("shiftguard", *ABLATIONS))
        & (raw.calibration == "group-prequential")
        & (raw.selection == "risk")
        & (raw.family != "clean")
        & np.isclose(raw.rate, 0.3)
    ]
    rng = np.random.default_rng(args.seed)
    primary_result = compare(primary, PRIMARY_MODELS, ("mae", "aurc"), args.bootstrap, rng)
    ablation_result = compare(ablations, ABLATIONS, ("mae", "aurc"), args.bootstrap, rng)
    for name, result in (("stress_paired_tests", primary_result), ("ablation_paired_tests", ablation_result)):
        result.to_csv(args.output / f"{name}.csv", index=False)
        (args.output / f"{name}.tex").write_text(
            result.to_latex(index=False, float_format="%.4f", escape=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
