from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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
    "shiftguard": "ShiftGuard",
    "staeformer": "STAEformer",
    "stgcn": "STGCN",
    "stid": "STID",
    "shiftguard_noreliability": "No reliability conditioning",
    "shiftguard_staticgraph": "Static graph only",
    "shiftguard_adaptivegraph": "Adaptive graph only",
    "shiftguard_norouter": "No regime router",
    "shiftguard_noaugmentation": "No outage augmentation",
    "shiftguard_independentaugmentation": "Independent masking only",
}

ABLATION_MODELS = {
    "shiftguard_noreliability",
    "shiftguard_staticgraph",
    "shiftguard_adaptivegraph",
    "shiftguard_norouter",
    "shiftguard_noaugmentation",
    "shiftguard_independentaugmentation",
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
        "graphwavenet", "agcrn", "stid", "staeformer", "shiftguard",
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
        & summary["model"].isin({"shiftguard", *ABLATION_MODELS})
    ]
    ablation = (
        ablation_source.groupby(["dataset", "model"])
        .agg(mae=("mae", "mean"), aurc=("aurc", "mean"), retained=("retained", "mean"))
        .reset_index()
        .rename(columns={"dataset": "Dataset", "model": "Variant", "mae": "MAE", "aurc": "AURC", "retained": "Retained"})
    )
    ablation["Variant"] = ablation["Variant"].replace(MODEL_LABELS)
    ablation_latex = ablation.to_latex(index=False, float_format="%.3f", escape=True)
    (destination / "ablation_main.tex").write_text(ablation_latex, encoding="utf-8")
    if paper.is_dir():
        (paper / "generated_ablation.tex").write_text(ablation_latex, encoding="utf-8")

    calibration_source = summary[
        (summary["model"] == "shiftguard")
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


def aggregate_efficiency(destination: Path, paper: Path) -> None:
    path = destination / "efficiency.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    selected = frame[frame.model.isin({"graphwavenet", "dcrnn", "stid", "staeformer", "shiftguard"})]
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
    aggregate_efficiency(destination, paper)


if __name__ == "__main__":
    main()
