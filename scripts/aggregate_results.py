from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from shiftguard.registry import (
    LEGACY_ABLATIONS,
    LEGACY_MODEL,
    REFERENCE_ABLATIONS,
    REFERENCE_MODEL,
)

MODEL_LABELS = {
    "agcrn": "AGCRN",
    "dcrnn": "DCRNN",
    "graphwavenet": "Graph WaveNet",
    "gru": "GRU",
    "historical_average": "Historical average",
    "linear": "Linear",
    "mlp": "MLP",
    "persistence": "Persistence",
    "ridge_ar": "Ridge AR",
    REFERENCE_MODEL: "ShiftGuard",
    LEGACY_MODEL: "ShiftGuard + reliability weighting",
    "staeformer": "STAEformer",
    "stgcn": "STGCN",
    "stid": "STID",
    "shiftguard_staticgraph": "Static graph only",
    "shiftguard_adaptivegraph": "Adaptive graph only",
    "shiftguard_norouter": "No regime router",
    "shiftguard_noaugmentation": "No outage augmentation",
    "shiftguard_independentaugmentation": "Independent masking only",
}
OUTAGE_AUGMENTED_SUFFIX = "_outageaug"
for _name in ("mlp", "gru", "dcrnn", "graphwavenet", "agcrn", "stid", "staeformer"):
    MODEL_LABELS[f"{_name}{OUTAGE_AUGMENTED_SUFFIX}"] = f"{MODEL_LABELS[_name]} + outage aug."
    MODEL_LABELS[f"{_name}{OUTAGE_AUGMENTED_SUFFIX}_laplace"] = (
        f"{MODEL_LABELS[_name]} + outage aug. + Laplace"
    )

for _flag, _label in (
    ("staticgraph", "Static graph only"),
    ("adaptivegraph", "Adaptive graph only"),
    ("norouter", "No regime router"),
    ("noaugmentation", "No outage augmentation"),
    ("independentaugmentation", "Independent masking only"),
):
    MODEL_LABELS[f"{REFERENCE_MODEL}_{_flag}"] = _label
# Every ShiftGuard variant except the reference is an ablation.
ABLATION_MODELS = (set(REFERENCE_ABLATIONS) | set(LEGACY_ABLATIONS)) - {REFERENCE_MODEL}
LEGACY_LABELS = {
    LEGACY_MODEL: "Legacy ShiftGuard (reference + reliability weighting)",
    "shiftguard_noreliability": "No reliability weighting (= reference)",
    "shiftguard_staticgraph": "Static graph only",
    "shiftguard_adaptivegraph": "Adaptive graph only",
    "shiftguard_norouter": "No regime router",
    "shiftguard_noaugmentation": "No outage augmentation",
    "shiftguard_independentaugmentation": "Independent masking only",
}


def aggregate_stress(destination: Path, paper: Path) -> None:
    rows: list[dict[str, object]] = []
    for path in Path("artifacts").glob("*/*/seed_*/stress_rate_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("results", payload)
        calibration = payload.get("calibration", "group-prequential")
        selection = payload.get("selection", "risk")
        file_rate = float(path.stem.removeprefix("stress_rate_").split("_")[0])
        rate = float(payload.get("stress_rate", file_rate))
        dataset, model, seed_dir = path.parts[-4], path.parts[-3], path.parts[-2]
        seed = int(seed_dir.removeprefix("seed_"))
        for family, result in results.items():
            average = result["point"]["average"]
            regimes = result.get("regimes", {})
            interval_groups = result.get("interval_groups", {})
            regime_maes = [metrics["mae"] for metrics in regimes.values()]
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "rate": rate,
                    "calibration": calibration,
                    "selection": selection,
                    "family": family,
                    "mae": average["mae"],
                    "rmse": average["rmse"],
                    "mape": average["mape"],
                    "worst_regime_mae": max(regime_maes) if regime_maes else float("nan"),
                    "coverage": result["interval"]["coverage"],
                    "worst_node_horizon_coverage": interval_groups.get(
                        "worst_node_horizon_coverage", float("nan")
                    ),
                    "worst_regime_coverage": interval_groups.get(
                        "worst_regime_coverage", float("nan")
                    ),
                    "width": result["interval"]["mean_width"],
                    "winkler": result["interval"]["winkler"],
                    "aurc": result["aurc"],
                    "retained": result["selective"]["coverage"],
                    "selective_mae": result["selective"]["mae"],
                    "failure_rate": result["selective"]["failure_rate_5mph"],
                }
            )
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "stress_all_runs.csv", index=False)
    grouping = ["dataset", "model", "rate", "calibration", "selection", "family"]
    summary = (
        frame.groupby(grouping, dropna=False)
        .agg(
            mae=("mae", "mean"),
            mae_sd=("mae", "std"),
            worst_regime_mae=("worst_regime_mae", "mean"),
            coverage=("coverage", "mean"),
            worst_node_horizon_coverage=("worst_node_horizon_coverage", "mean"),
            worst_regime_coverage=("worst_regime_coverage", "mean"),
            width=("width", "mean"),
            winkler=("winkler", "mean"),
            aurc=("aurc", "mean"),
            retained=("retained", "mean"),
            selective_mae=("selective_mae", "mean"),
            failure_rate=("failure_rate", "mean"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(destination / "stress_summary.csv", index=False)
    primary_models = {
        "persistence", "linear", "mlp", "gru", "stgcn", "dcrnn",
        "graphwavenet", "agcrn", "stid", "staeformer", REFERENCE_MODEL,
    }
    main_source = summary[
        (summary["rate"] == 0.3)
        & (summary["calibration"] == "group-prequential")
        & (summary["selection"] == "risk")
        & (summary["family"] != "clean")
        & summary["model"].isin(primary_models)
        & (summary["seeds"] >= 5)
    ]
    main = (
        main_source.groupby(["dataset", "model"])
        .agg(
            mae=("mae", "mean"), coverage=("coverage", "mean"),
            worst_node_horizon_coverage=("worst_node_horizon_coverage", "mean"),
            worst_regime_coverage=("worst_regime_coverage", "mean"),
            aurc=("aurc", "mean"),
            retained=("retained", "mean"), failure_rate=("failure_rate", "mean"),
            seeds=("seeds", "min"),
        )
        .reset_index()
    )
    if main.empty:
        return
    main = main[
        ["dataset", "model", "mae", "coverage", "worst_node_horizon_coverage",
         "worst_regime_coverage", "aurc", "retained", "failure_rate", "seeds"]
    ].rename(
        columns={
            "dataset": "Dataset",
            "model": "Model",
            "mae": "MAE",
            "coverage": "PI coverage",
            "worst_node_horizon_coverage": "Worst N-H cov.",
            "worst_regime_coverage": "Worst regime cov.",
            "aurc": "AURC",
            "retained": "Retained",
            "failure_rate": "Failure rate",
            "seeds": "Seeds",
        }
    )
    main["Model"] = main["Model"].replace(MODEL_LABELS)
    latex = main.to_latex(index=False, float_format="%.3f", na_rep="--", escape=True)
    (destination / "stress_main.tex").write_text(latex, encoding="utf-8")
    if paper.is_dir():
        (paper / "generated_stress.tex").write_text(latex, encoding="utf-8")

    ablation_source = summary[
        (summary["rate"] == 0.3)
        & (summary["calibration"] == "group-prequential")
        & (summary["selection"] == "risk")
        & (summary["family"] != "clean")
    ]
    for name, members, labels in (
        ("ablation", {REFERENCE_MODEL, *REFERENCE_ABLATIONS}, {
            **MODEL_LABELS, LEGACY_MODEL: "+ reliability weighting (legacy model)"}),
        ("ablation_legacy", {LEGACY_MODEL, *LEGACY_ABLATIONS}, LEGACY_LABELS),
    ):
        selected = ablation_source[ablation_source["model"].isin(members)]
        if selected.empty:
            continue
        ablation = (
            selected.groupby(["dataset", "model"])
            .agg(mae=("mae", "mean"), aurc=("aurc", "mean"),
                 retained=("retained", "mean"), seeds=("seeds", "min"))
            .reset_index()
            .rename(columns={"dataset": "Dataset", "model": "Variant", "mae": "MAE",
                             "aurc": "AURC", "retained": "Retained", "seeds": "Seeds"})
        )
        ablation["Variant"] = ablation["Variant"].replace(labels)
        ablation_latex = ablation.to_latex(index=False, float_format="%.3f", escape=True)
        (destination / f"{name}_main.tex").write_text(ablation_latex, encoding="utf-8")
        if paper.is_dir():
            (paper / f"generated_{name}.tex").write_text(ablation_latex, encoding="utf-8")

    calibration_source = summary[
        (summary["model"] == REFERENCE_MODEL)
        & (summary["rate"] == 0.3)
        & (summary["family"] != "clean")
    ]
    calibration = (
        calibration_source.groupby(["dataset", "calibration", "selection"])
        .agg(coverage=("coverage", "mean"), retained=("retained", "mean"), failure_rate=("failure_rate", "mean"))
        .reset_index()
        .rename(columns={"dataset": "Dataset", "calibration": "Calibration", "selection": "Selection", "coverage": "PI coverage", "retained": "Retained", "failure_rate": "Failure rate"})
    )
    calibration_latex = calibration.to_latex(index=False, float_format="%.3f", escape=True)
    (destination / "calibration_main.tex").write_text(calibration_latex, encoding="utf-8")
    if paper.is_dir():
        (paper / "generated_calibration.tex").write_text(calibration_latex, encoding="utf-8")


def aggregate_fair_comparison(destination: Path, paper: Path) -> None:
    """Compare each baseline with and without the outage-augmentation recipe.

    Only baselines that have an outage-augmented counterpart are reported, so
    the table isolates the architecture from the training-time masking.
    """

    summary_path = destination / "stress_summary.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    variants = {
        model for model in summary["model"].unique() if OUTAGE_AUGMENTED_SUFFIX in model
    }
    if not variants:
        return
    bases = {model.split(OUTAGE_AUGMENTED_SUFFIX)[0] for model in variants}
    models = {REFERENCE_MODEL, *bases, *variants}
    selected = summary[
        (summary["rate"] == 0.3)
        & (summary["calibration"] == "group-prequential")
        & (summary["selection"] == "risk")
        & summary["model"].isin(models)
    ]
    stressed = (
        selected[selected["family"] != "clean"]
        .groupby(["dataset", "model"])
        .agg(
            stress_mae=("mae", "mean"), aurc=("aurc", "mean"),
            retained=("retained", "mean"), failure_rate=("failure_rate", "mean"),
            seeds=("seeds", "min"),
        )
    )
    clean = (
        selected[selected["family"] == "clean"]
        .groupby(["dataset", "model"])
        .agg(clean_mae=("mae", "mean"))
    )
    fair = clean.join(stressed, how="right").reset_index()
    fair["augmented"] = fair["model"].str.contains(OUTAGE_AUGMENTED_SUFFIX) | (
        fair["model"] == REFERENCE_MODEL
    )
    fair = fair.sort_values(["dataset", "stress_mae"])
    fair.to_csv(destination / "fair_comparison.csv", index=False)
    display = fair[
        ["dataset", "model", "clean_mae", "stress_mae", "aurc", "retained",
         "failure_rate", "seeds"]
    ].rename(
        columns={
            "dataset": "Dataset", "model": "Model", "clean_mae": "Clean MAE",
            "stress_mae": "Stress MAE", "aurc": "AURC", "retained": "Retained",
            "failure_rate": "Failure rate", "seeds": "Seeds",
        }
    )
    display["Model"] = display["Model"].replace(MODEL_LABELS)
    latex = display.to_latex(index=False, float_format="%.3f", escape=True, na_rep="--")
    (destination / "fair_comparison.tex").write_text(latex, encoding="utf-8")
    if paper.is_dir():
        (paper / "generated_fair_comparison.tex").write_text(latex, encoding="utf-8")


NATURAL_STRATA = {
    "complete": "Complete inputs",
    "sensor_outage_low": "Dead sensors, <=5% missing",
    "sensor_outage_high": "Dead sensors, >5% missing",
    "network_wide": "Network-wide dropout",
}


def aggregate_natural_missingness(destination: Path, paper: Path) -> None:
    """Metrics stratified by the dataset's own missingness (no synthetic masks)."""

    rows: list[dict[str, object]] = []
    for path in Path("artifacts").glob("*/*/seed_*/natural_missingness.json"):
        dataset, model, seed_dir = path.parts[-4], path.parts[-3], path.parts[-2]
        payload = json.loads(path.read_text(encoding="utf-8"))
        for stratum, result in payload["results"].items():
            rows.append(
                {"dataset": dataset, "model": model,
                 "seed": int(seed_dir.removeprefix("seed_")), "stratum": stratum, **result}
            )
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "natural_missingness_all_runs.csv", index=False)
    summary = (
        frame.groupby(["dataset", "model", "stratum"])
        .agg(
            origins=("origins", "first"), mae=("mae", "mean"), mae_sd=("mae", "std"),
            dead_sensor_mae=("dead_sensor_mae", "mean"), coverage=("coverage", "mean"),
            retained=("retained", "mean"), failure_rate=("failure_rate_5mph", "mean"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(destination / "natural_missingness_summary.csv", index=False)
    shown = {
        REFERENCE_MODEL, f"{REFERENCE_MODEL}_noaugmentation", LEGACY_MODEL,
        "shiftguard_noaugmentation", "staeformer", "stid", "dcrnn",
        "graphwavenet",
        "staeformer_outageaug", "staeformer_outageaug_laplace",
    }
    table = summary[
        summary["model"].isin(shown)
        & summary["stratum"].isin(NATURAL_STRATA)
        & (summary["seeds"] >= 5)
    ].copy()
    if table.empty:
        return
    table["stratum"] = pd.Categorical(table["stratum"], list(NATURAL_STRATA), ordered=True)
    table = table.sort_values(["dataset", "stratum", "mae"])
    table["stratum"] = table["stratum"].astype(str).replace(NATURAL_STRATA)
    table["model"] = table["model"].replace(
        {
            **MODEL_LABELS,
            f"{REFERENCE_MODEL}_noaugmentation": "ShiftGuard w/o outage aug.",
            "shiftguard_noaugmentation": "Legacy ShiftGuard w/o outage aug.",
        }
    )
    display = table[
        ["dataset", "stratum", "origins", "model", "mae", "dead_sensor_mae",
         "coverage", "retained", "failure_rate"]
    ].rename(
        columns={
            "dataset": "Dataset", "stratum": "Input missingness", "origins": "Origins",
            "model": "Model", "mae": "MAE", "dead_sensor_mae": "Dead-sensor MAE",
            "coverage": "PI coverage", "retained": "Retained", "failure_rate": "Failure rate",
        }
    )
    latex = display.to_latex(index=False, float_format="%.3f", escape=True, na_rep="--")
    (destination / "natural_missingness.tex").write_text(latex, encoding="utf-8")
    if paper.is_dir():
        (paper / "generated_natural_missingness.tex").write_text(latex, encoding="utf-8")


def aggregate_efficiency(destination: Path, paper: Path) -> None:
    path = destination / "efficiency.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    # The reference model is benchmarked separately, on an idle GPU.
    reference = destination / "efficiency_reference.csv"
    if reference.exists():
        extra = pd.read_csv(reference)
        frame = pd.concat([frame[~frame.model.isin(extra.model.unique())], extra])
    selected = frame[frame.model.isin({"graphwavenet", "dcrnn", "stid", "staeformer", REFERENCE_MODEL})]
    batch_one = selected[selected.batch_size == 1][
        ["dataset", "model", "parameters", "profiler_flops_per_sample", "median_latency_ms"]
    ]
    batch_eight = selected[selected.batch_size == 8][
        ["dataset", "model", "throughput_samples_per_second", "peak_allocated_memory_mb"]
    ]
    display = batch_one.merge(batch_eight, on=["dataset", "model"]).rename(
        columns={"dataset": "Dataset", "model": "Model", "parameters": "Parameters",
                 "profiler_flops_per_sample": "FLOPs/sample", "median_latency_ms": "Latency B1 (ms)",
                 "throughput_samples_per_second": "Throughput B8", "peak_allocated_memory_mb": "Memory B8 (MiB)"}
    )
    display["Model"] = display["Model"].replace(MODEL_LABELS)
    latex = display.to_latex(index=False, float_format="%.2f", escape=True)
    (destination / "efficiency_main.tex").write_text(latex, encoding="utf-8")
    if paper.is_dir():
        (paper / "generated_efficiency.tex").write_text(latex, encoding="utf-8")


def main() -> None:
    rows: list[dict[str, object]] = []
    for path in Path("artifacts").glob("*/*/seed_*/metrics.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        average = result["metrics"]["average"]
        rows.append(
            {
                "dataset": result["dataset"],
                "model": result["model"],
                "seed": result["seed"],
                "mae": average["mae"],
                "rmse": average["rmse"],
                "mape": average["mape"],
                "parameters": result["parameters"],
                "training_seconds": result["training_seconds"],
            }
        )
    if not rows:
        raise SystemExit("no completed metrics found")
    frame = pd.DataFrame(rows)
    destination = Path("artifacts/summary")
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "all_runs.csv", index=False)
    summary = (
        frame.groupby(["dataset", "model"])
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            mape_mean=("mape", "mean"),
            parameters=("parameters", "first"),
            training_seconds=("training_seconds", "mean"),
            seeds=("seed", "count"),
        )
        .reset_index()
    )
    summary.to_csv(destination / "headline.csv", index=False)
    deterministic = {"historical_average", "persistence", "ridge_ar"}
    confirmed = summary[
        (summary["seeds"] >= 5) | summary["model"].isin(deterministic)
    ].copy()
    confirmed = confirmed[~confirmed["model"].isin(ABLATION_MODELS)]
    confirmed = confirmed[~confirmed["model"].str.contains(OUTAGE_AUGMENTED_SUFFIX)]
    confirmed = confirmed.sort_values(["dataset", "mae_mean"])
    display = confirmed.rename(
        columns={
            "dataset": "Dataset",
            "model": "Model",
            "mae_mean": "MAE",
            "mae_std": "SD",
            "rmse_mean": "RMSE",
            "mape_mean": "MAPE (pct.)",
            "parameters": "Parameters",
            "training_seconds": "Train (s)",
            "seeds": "Seeds",
        }
    )
    display["Model"] = display["Model"].replace(MODEL_LABELS)
    display["SD"] = display["SD"].map(
        lambda value: "--" if pd.isna(value) else f"{value:.3f}"
    )
    latex = display.to_latex(index=False, float_format="%.3f", escape=True)
    (destination / "headline.tex").write_text(latex, encoding="utf-8")
    paper = Path("paper")
    if paper.is_dir():
        (paper / "generated_headline.tex").write_text(latex, encoding="utf-8")

    eda_rows: list[dict[str, object]] = []
    for path in sorted(Path("artifacts/eda").glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        eda_rows.append(
            {
                "Dataset": item["dataset"],
                "Timestamps": item["timestamps"],
                "Sensors": item["sensors"],
                "Missing (pct.)": 100 * item["missing_fraction"],
                "Mean speed": item["mean_speed"],
                "Std. speed": item["std_speed"],
                "Lag-1 corr.": item["autocorrelation"]["lag_1_5min"],
                "Lag-288 corr.": item["autocorrelation"]["lag_288_1day"],
            }
        )
    if eda_rows:
        eda = pd.DataFrame(eda_rows)
        eda.to_csv(destination / "datasets.csv", index=False)
        eda_latex = eda.to_latex(index=False, float_format="%.3f", escape=True)
        (destination / "datasets.tex").write_text(eda_latex, encoding="utf-8")
        if paper.is_dir():
            (paper / "generated_datasets.tex").write_text(eda_latex, encoding="utf-8")

    reported_path = Path("docs/reported_results.csv")
    if reported_path.exists():
        reported = pd.read_csv(reported_path)
        context_rows: list[dict[str, str]] = []
        for (dataset, model), group in reported.groupby(["dataset", "model"], sort=True):
            values = {row.horizon: row.mae for row in group.itertuples()}
            context_rows.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    "MAE at 15 min": f"{values['15min']:.2f}" if "15min" in values else "--",
                    "MAE at 30 min": f"{values['30min']:.2f}" if "30min" in values else "--",
                    "MAE at 60 min": f"{values['60min']:.2f}" if "60min" in values else "--",
                    "Average MAE": f"{values['average']:.2f}" if "average" in values else "--",
                }
            )
        context = pd.DataFrame(context_rows)
        context.to_csv(destination / "reported_context.csv", index=False)
        context_latex = context.to_latex(index=False, escape=True)
        (destination / "reported_context.tex").write_text(context_latex, encoding="utf-8")
        if paper.is_dir():
            (paper / "generated_reported_context.tex").write_text(
                context_latex, encoding="utf-8"
            )
    aggregate_stress(destination, paper)
    aggregate_fair_comparison(destination, paper)
    aggregate_natural_missingness(destination, paper)
    aggregate_efficiency(destination, paper)


if __name__ == "__main__":
    main()
