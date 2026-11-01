"""Build the manuscript's deployment tables and figures from artifacts/deployment.

Rerun as seeds finish: every number comes from finished scenario results, each
comparison uses only seeds shared by its models, and every table shows its seed
count. Nothing here trains, selects, or tunes a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aggregate_deployment import (
    OUTAGES,
    PRIMARY,
    confirmatory_for_gating,
    flatten,
    load_runs,
    paired_intervals,
)

from shiftguard.registry import LEGACY_MODEL, REFERENCE_MODEL

PAPER = Path("paper")
FIGURES = PAPER / "figures"
SUMMARY = Path("artifacts/summary")
DATASETS = ("METR-LA", "PEMS-BAY")
SEEDS = 5
ARM = "feasible"
BASELINES = ("stid", "dcrnn", "graphwavenet", "agcrn", "staeformer")
FORMS = (("", "As published"), ("_outageaug", "+ outage aug."), ("_outageaug_laplace", "+ outage aug. + Laplace"))
LABELS = {
    REFERENCE_MODEL: "ShiftGuard",
    LEGACY_MODEL: "+ reliability weighting",
    "stid": "STID",
    "dcrnn": "DCRNN",
    "graphwavenet": "Graph WaveNet",
    "agcrn": "AGCRN",
    "staeformer": "STAEformer",
    f"{REFERENCE_MODEL}_noaugmentation": "No outage augmentation",
    f"{REFERENCE_MODEL}_independentaugmentation": "Independent masking only",
    f"{REFERENCE_MODEL}_nomaskinput": "No mask input",
    f"{REFERENCE_MODEL}_norouter": "No regime router",
    f"{REFERENCE_MODEL}_staticgraph": "Static graph only",
    f"{REFERENCE_MODEL}_adaptivegraph": "Adaptive graph only",
}
ABLATIONS = tuple(name for name in LABELS if name.startswith(f"{REFERENCE_MODEL}_")) + (LEGACY_MODEL,)
SCENARIO_LABELS = {
    "native": "Native",
    "trace_x1": "Trace x1",
    "trace_x2": "Trace x2",
    "sensor_s30": "Sensor 30%",
    "region_s30": "Region 30%",
    "hub_s15": "Hub 15%",
}
# Fixed identity colors (validated default palette, slots in order).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#333333", "#6b6b66", "#e4e3df"
plt.rcParams.update(
    {
        "font.size": 8,
        "font.family": "serif",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


def summarize(frame: pd.DataFrame, dataset: str, model: str, seeds=None, scenarios=OUTAGES) -> dict | None:
    """Mean over scenarios then seeds, using seeds that finished all primary scenarios."""

    rows = frame[(frame.dataset == dataset) & (frame.model == model)]
    finished = rows[rows.scenario.isin(PRIMARY)].groupby("seed").scenario.nunique()
    usable = set(finished[finished == len(PRIMARY)].index)
    if seeds is not None:
        usable &= set(seeds)
    if not usable:
        return None
    chosen = rows[rows.seed.isin(usable) & rows.scenario.isin(scenarios)]
    numeric = chosen.drop(columns=["dataset", "model", "scenario"]).groupby("seed").mean(numeric_only=True)
    result = numeric.mean(numeric_only=True).to_dict()
    result["seeds"] = len(usable)
    return result


def finished_seeds(frame: pd.DataFrame, dataset: str, model: str) -> set[int]:
    rows = frame[(frame.dataset == dataset) & (frame.model == model) & frame.scenario.isin(PRIMARY)]
    counts = rows.groupby("seed").scenario.nunique()
    return set(counts[counts == len(PRIMARY)].index)


def write_table(table: pd.DataFrame, name: str, formats: dict | None = None) -> None:
    latex = table.to_latex(
        index=False, float_format="%.3f", escape=True, na_rep="--", formatters=formats
    )
    (PAPER / f"generated_dep_{name}.tex").write_text(latex, encoding="utf-8")
    table.to_csv(SUMMARY / f"deployment_{name}.csv", index=False)


def main_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        block = []
        for model in (REFERENCE_MODEL, *BASELINES):
            outage = summarize(frame, dataset, model)
            native = summarize(frame, dataset, model, scenarios=("native",))
            if outage is None:
                continue
            block.append(
                {
                    "Dataset": dataset, "Model": LABELS[model], "Native MAE": native["mae"],
                    "Outage MAE": outage["mae"], "Blind MAE": outage["mae_blind"],
                    "PI cov.": outage[f"{ARM}|coverage"], "Blind cov.": outage[f"{ARM}|coverage_blind"],
                    "Retained": outage[f"{ARM}|retained"], "Failure": outage[f"{ARM}|failure_released"],
                    "Seeds": outage["seeds"],
                }
            )
        block.sort(key=lambda row: row["Outage MAE"])
        reference = summarize(frame, dataset, REFERENCE_MODEL)
        native = summarize(frame, dataset, REFERENCE_MODEL, scenarios=("native",))
        block.append(
            {
                "Dataset": dataset, "Model": "Historical profile (fallback)",
                "Native MAE": native["fallback_mae"], "Outage MAE": reference["fallback_mae"],
                "Blind MAE": reference["fallback_mae_blind"], "Seeds": reference["seeds"],
            }
        )
        rows += block
    return pd.DataFrame(rows)


def scenario_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        shared = finished_seeds(frame, dataset, REFERENCE_MODEL) & finished_seeds(frame, dataset, "staeformer")
        for scenario in PRIMARY:
            ref = summarize(frame, dataset, REFERENCE_MODEL, shared, (scenario,))
            other = summarize(frame, dataset, "staeformer", shared, (scenario,))
            rows.append(
                {
                    "Dataset": dataset, "Scenario": SCENARIO_LABELS[scenario],
                    "Unavail.": ref["cells_unavailable"],
                    "MAE SG": ref["mae"], "MAE STAE": other["mae"],
                    "Cov. SG": ref[f"{ARM}|coverage"], "Cov. STAE": other[f"{ARM}|coverage"],
                    "Ret. SG": ref[f"{ARM}|retained"], "Ret. STAE": other[f"{ARM}|retained"],
                    "Fail. SG": ref[f"{ARM}|failure_released"], "Fail. STAE": other[f"{ARM}|failure_released"],
                }
            )
    return pd.DataFrame(rows)


def recipe_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Every form of a baseline and the reference on seeds they all finished."""

    rows = []
    for dataset in DATASETS:
        for base in BASELINES:
            names = [(base + suffix, form) for suffix, form in FORMS]
            present = [(n, f) for n, f in names if finished_seeds(frame, dataset, n)]
            shared = finished_seeds(frame, dataset, REFERENCE_MODEL)
            for name, _ in present:
                shared &= finished_seeds(frame, dataset, name)
            if not shared:
                continue
            ref = summarize(frame, dataset, REFERENCE_MODEL, shared)
            for name, form in present:
                item = summarize(frame, dataset, name, shared)
                rows.append(
                    {
                        "Dataset": dataset, "Baseline": LABELS[base], "Form": form, "MAE": item["mae"],
                        "vs SG": item["mae"] - ref["mae"], "Gated MAE": item["gated_mae"],
                        "Retained": item[f"{ARM}|retained"], "Failure": item[f"{ARM}|failure_released"],
                        "Seeds": len(shared),
                    }
                )
    return pd.DataFrame(rows)


def ablation_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        reference_seeds = finished_seeds(frame, dataset, REFERENCE_MODEL)
        ref = summarize(frame, dataset, REFERENCE_MODEL)
        rows.append(
            {
                "Dataset": dataset, "Variant": "ShiftGuard (reference)", "MAE": ref["mae"], "Delta MAE": 0.0,
                "Blind MAE": ref["mae_blind"], "Retained": ref[f"{ARM}|retained"],
                "Failure": ref[f"{ARM}|failure_released"], "Seeds": ref["seeds"],
            }
        )
        for name in ABLATIONS:
            shared = reference_seeds & finished_seeds(frame, dataset, name)
            if not shared:
                continue
            base, item = (summarize(frame, dataset, m, shared) for m in (REFERENCE_MODEL, name))
            rows.append(
                {
                    "Dataset": dataset, "Variant": LABELS[name], "MAE": item["mae"],
                    "Delta MAE": item["mae"] - base["mae"], "Blind MAE": item["mae_blind"],
                    "Retained": item[f"{ARM}|retained"], "Failure": item[f"{ARM}|failure_released"],
                    "Seeds": len(shared),
                }
            )
    return pd.DataFrame(rows)


def arm_table(frame: pd.DataFrame) -> pd.DataFrame:
    names = (("oracle_feedback", "Oracle feedback"), ("feasible", "Feasible"), ("feasible_thinned", "Feasible, thinned"))
    rows = []
    for dataset in DATASETS:
        item = summarize(frame, dataset, REFERENCE_MODEL)
        for arm, label in names:
            rows.append(
                {
                    "Dataset": dataset, "Calibration": label, "PI cov.": item[f"{arm}|coverage"],
                    "Blind cov.": item[f"{arm}|coverage_blind"], "Worst regime": item[f"{arm}|worst_regime_coverage"],
                    "Retained": item[f"{arm}|retained"], "Failure": item[f"{arm}|failure_released"],
                    "Failure AUROC": item[f"{arm}|failure_auroc"], "Days over limit": item[f"{arm}|days_over_risk_limit"],
                    "Certified": item[f"{arm}|certified_blocks"],
                }
            )
    return pd.DataFrame(rows)


def origin_always_release(item: dict, fallback: float) -> float:
    """Origin-weighted MAE of always releasing, the correct comparator for the policy.

    ``policy_mae_random = c * m + (1 - c) * F``; ``F`` depends only on the data.
    """

    coverage = item["retained"]
    return (item["policy_mae_random"] - (1 - coverage) * fallback) / coverage


def fallback_origin_mae() -> dict[str, float]:
    """Cached origin-weighted MAE of the historical profile per dataset (CPU only)."""

    cache = SUMMARY / "deployment_fallback_origin_mae.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from evaluate_deployment import calendar

    from shiftguard.calibration import finite_mean
    from shiftguard.outages import HISTORY, HORIZON, historical_profile

    values = {}
    for dataset in DATASETS:
        root = Path("data/processed") / dataset
        with np.load(root / "scaler.npz") as archive:
            mean, std = archive["mean"], archive["std"]
        with np.load(root / "train.npz") as archive:
            train_x = archive["x"][:, 0]
        time_train, day_train = calendar(train_x[:, None], len(train_x))
        table = historical_profile(train_x[..., 0] * std + mean, train_x[..., 1] > 0, time_train, day_train)
        del train_x
        with np.load(root / "test.npz") as archive:
            x0, y = archive["x"][:1], archive["y"]
        origins = len(y)
        time_test, day_test = calendar(x0, origins + HISTORY + HORIZON - 1)
        steps = np.arange(origins)[:, None] + HISTORY + np.arange(HORIZON)[None]
        profile = table[day_test[steps], time_test[steps]]
        target = np.where(y[..., 1] > 0, y[..., 0] * std + mean, np.nan)
        values[dataset] = float(np.nanmean(finite_mean(np.abs(target - profile), (1, 2))))
    cache.write_text(json.dumps(values, indent=2))
    return values


def hypotheses(runs: dict) -> pd.DataFrame:
    """H1--H4 on seeds that are confirmatory for gating, reported per seed."""

    fallback = fallback_origin_mae()
    rows = []
    for dataset in DATASETS:
        seeds = sorted(
            s for (d, m, s), r in runs.items()
            if d == dataset and m == REFERENCE_MODEL and confirmatory_for_gating(d, s)
            and all(c in r for c in OUTAGES)
        )
        per_seed = {"H1": [], "H2": [], "H3": [], "H4": []}
        means = {key: [] for key in per_seed}
        for seed in seeds:
            record = runs[(dataset, REFERENCE_MODEL, seed)]
            h = {key: [] for key in per_seed}
            for scenario in OUTAGES:
                item = record[scenario]
                blind, fresh = item["evidence"]["buckets"]["blind"], item["evidence"]["buckets"]["fresh"]
                arm = item["arms"][ARM]
                if blind["mae_model"] is not None:
                    h["H1"].append((blind["mae_gated"], blind["mae_model"]))
                if blind["coverage"] is not None and arm["coverage_blind"] is not None:
                    h["H3"].append((abs(blind["coverage"] - 0.9), abs(arm["coverage_blind"] - 0.9)))
                h["H2"].append((fresh["mae_gated"], fresh["mae_model"]))
                always = origin_always_release(arm, fallback[dataset])
                h["H4"].append((arm["policy_mae_fallback"], always))
            for key, pairs in h.items():
                values = np.mean(pairs, axis=0)
                means[key].append(values)
                per_seed[key].append(values[0] - values[1])
        statements = {
            "H1": ("Gated vs model MAE, blind sensors", "lower"),
            "H2": ("Gated vs model MAE, fresh sensors (limit +1%)", "limit"),
            "H3": ("Blind coverage gap to 0.90, shadow vs feasible", "lower"),
            "H4": ("Policy vs always-release MAE", "lower"),
        }
        for key, (statement, rule) in statements.items():
            a, b = np.mean(means[key], axis=0)
            diffs = np.array(per_seed[key])
            if rule == "limit":
                ratios = [x / y - 1 for x, y in means[key]]
                verdict = "Supported" if max(ratios) <= 0.01 else "Not supported"
            else:
                verdict = "Supported" if (diffs < 0).all() else ("Mixed" if (diffs < 0).any() else "Not supported")
            rows.append(
                {
                    "Dataset": dataset, "H": key, "Comparison": statement, "Proposed": a, "Comparator": b,
                    "Difference": a - b, "Seed range": f"[{diffs.min():.3f}, {diffs.max():.3f}]",
                    "Seeds": len(seeds), "Outcome": verdict,
                }
            )
    return pd.DataFrame(rows)


def bars(axis, groups, series, width=0.8):
    """Grouped bars with a surface gap; ``series`` is ``[(label, color, values)]``."""

    step = width / len(series)
    for index, (label, color, values) in enumerate(series):
        positions = np.arange(len(groups)) + (index - (len(series) - 1) / 2) * step
        axis.bar(positions, values, step, label=label, color=color, edgecolor="white", linewidth=0.8)
    axis.set_xticks(np.arange(len(groups)), groups)
    axis.grid(axis="x", visible=False)


def figure_scenarios(frame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for axis, dataset in zip(axes, DATASETS):
        shared = finished_seeds(frame, dataset, REFERENCE_MODEL) & finished_seeds(frame, dataset, "staeformer")
        values = {
            model: [summarize(frame, dataset, model, shared, (s,))["mae"] for s in PRIMARY]
            for model in (REFERENCE_MODEL, "staeformer")
        }
        profile = [summarize(frame, dataset, REFERENCE_MODEL, shared, (s,))["fallback_mae"] for s in PRIMARY]
        bars(axis, [SCENARIO_LABELS[s] for s in PRIMARY], [
            ("ShiftGuard", BLUE, values[REFERENCE_MODEL]), ("STAEformer", ORANGE, values["staeformer"]),
            ("Historical profile", MUTED, profile),
        ])
        axis.set_title(dataset, fontsize=9)
        axis.tick_params(axis="x", labelrotation=30, labelsize=7)
    axes[0].set_ylabel("MAE (mph)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.32), ncols=3)
    fig.savefig(FIGURES / "dep_scenarios.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_policy(frame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    models = ((REFERENCE_MODEL, BLUE), ("staeformer", ORANGE), ("stid", AQUA), ("dcrnn", YELLOW))
    for axis, dataset in zip(axes, DATASETS):
        points = []
        top = max(12.0, 110 * max(summarize(frame, dataset, m)[f"{ARM}|failure_released"] for m, _ in models))
        axis.set_ylim(0, top)
        for model, color in models:
            item = summarize(frame, dataset, model)
            x, y = 100 * item[f"{ARM}|retained"], 100 * item[f"{ARM}|failure_released"]
            axis.scatter(x, y, s=34, color=color, edgecolor="white", linewidth=0.8, zorder=3, label=LABELS[model])
            points.append((x, y, LABELS[model]))
        # Direct labels; models at the same spot share one so none is hidden.
        while points:
            x, y, name = points.pop()
            near = [p for p in points if abs(p[0] - x) < 4 and abs(p[1] - y) < 0.06 * top]
            points = [p for p in points if p not in near]
            text = " / ".join([p[2] for p in near] + [name])
            left = x > 50
            axis.annotate(text, (x, y), xytext=(-6 if left else 6, 0), textcoords="offset points",
                          ha="right" if left else "left", va="center", fontsize=7)
        axis.axhline(10, color=MUTED, linewidth=0.9, linestyle="--", zorder=2)
        axis.annotate("10% risk limit", (2, 10), xytext=(0, 2), textcoords="offset points", fontsize=7, color=MUTED)
        axis.set_xlim(0, 104)
        axis.set_xlabel("Forecasts released (%)")
        axis.set_title(dataset, fontsize=9)
    axes[0].set_ylabel("Failure among released (%)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.2), ncols=4)
    fig.savefig(FIGURES / "dep_policy.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_recipe(recipe: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    forms = [form for _, form in FORMS]
    for axis, dataset in zip(axes, DATASETS):
        block = recipe[recipe.Dataset == dataset]
        names = [LABELS[b] for b in BASELINES if LABELS[b] in set(block.Baseline)]
        series = []
        for form, color in zip(forms, (ORANGE, AQUA, YELLOW)):
            lookup = block[block.Form == form].set_index("Baseline")["vs SG"]
            series.append((form, color, [lookup.get(name, np.nan) for name in names]))
        bars(axis, names, series)
        axis.axhline(0, color=BLUE, linewidth=1.2, label="ShiftGuard")
        axis.set_title(dataset, fontsize=9)
        axis.tick_params(axis="x", labelsize=7, labelrotation=25)
    axes[0].set_ylabel("MAE minus ShiftGuard (mph)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.24), ncols=4)
    fig.savefig(FIGURES / "dep_recipe.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_ablation(ablation: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
    order = [LABELS[name] for name in ABLATIONS]
    for axis, dataset in zip(axes, DATASETS):
        lookup = ablation[ablation.Dataset == dataset].set_index("Variant")["Delta MAE"]
        values = [lookup.get(name, np.nan) for name in order]
        axis.barh(np.arange(len(order)), values, 0.62, color=BLUE, edgecolor="white", linewidth=0.8)
        axis.axvline(0, color=MUTED, linewidth=0.9)
        axis.set_yticks(np.arange(len(order)), order)
        axis.grid(axis="y", visible=False)
        axis.set_title(dataset, fontsize=9)
        axis.set_xlabel("Change in outage MAE (mph)")
    axes[0].invert_yaxis()  # The panels share this axis, so flip it once.
    fig.tight_layout()
    fig.savefig(FIGURES / "dep_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_freshness(runs: dict) -> None:
    levels = ("fresh", "recent", "stale", "blind")
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 3.7), sharex=True)
    for column, dataset in enumerate(DATASETS):
        keys = [k for k in runs if k[:2] == (dataset, REFERENCE_MODEL) and confirmatory_for_gating(dataset, k[2])]
        collect = {field: {level: [] for level in levels} for field in ("mae_model", "mae_gated", "coverage")}
        for key in keys:
            for scenario in OUTAGES:
                for level in levels:
                    bucket = runs[key][scenario]["evidence"]["buckets"][level]
                    for field in collect:
                        if bucket[field] is not None:
                            collect[field][level].append(bucket[field])
        mean = {f: [np.mean(v[level]) if v[level] else np.nan for level in levels] for f, v in collect.items()}
        names = [level.capitalize() for level in levels]
        bars(axes[0, column], names, [("Model only", ORANGE, mean["mae_model"]), ("Evidence gated", BLUE, mean["mae_gated"])])
        axes[0, column].set_title(dataset, fontsize=9)
        bars(axes[1, column], names, [("Evidence-gated intervals", BLUE, mean["coverage"])], width=0.45)
        axes[1, column].axhline(0.9, color=MUTED, linewidth=0.9, linestyle="--")
        axes[1, column].set_ylim(0, 1)
    axes[0, 0].set_ylabel("MAE (mph)")
    axes[1, 0].set_ylabel("Interval coverage")
    axes[0, 0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.32), ncols=2)
    axes[1, 0].annotate("0.90 target", (-0.4, 0.9), xytext=(0, 2), textcoords="offset points", fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / "dep_freshness.pdf", bbox_inches="tight")
    plt.close(fig)


def status(frame: pd.DataFrame) -> list[str]:
    """Models still short of five seeds; shown in red in the draft until empty."""

    expected = (REFERENCE_MODEL, *BASELINES, *ABLATIONS)
    expected += tuple(f"{b}{s}" for b in BASELINES for s in ("_outageaug", "_outageaug_laplace"))
    missing = []
    for dataset in DATASETS:
        for model in expected:
            count = len(finished_seeds(frame, dataset, model))
            if count < SEEDS:
                missing.append(f"{dataset}/{model}:{count}")
    return missing


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    frame = flatten(runs)
    frame.to_csv(SUMMARY / "deployment_all_runs.csv", index=False)
    percent = lambda value: "--" if pd.isna(value) else f"{100 * value:.1f}"
    shares = {key: percent for key in ("Retained", "Failure")}

    write_table(main_table(frame), "main", shares)
    scenario_formats = {key: percent for key in ("Unavail.", "Ret. SG", "Ret. STAE", "Fail. SG", "Fail. STAE")}
    write_table(scenario_table(frame), "scenarios", scenario_formats)
    recipe = recipe_table(frame)
    write_table(recipe, "recipe", shares)
    ablation = ablation_table(frame)
    write_table(ablation, "ablation", shares)
    write_table(arm_table(frame), "arms", {**shares, "Days over limit": percent, "Certified": percent})
    write_table(hypotheses(runs), "hypotheses")
    contrasts = paired_intervals(runs, frame)
    contrasts.to_csv(SUMMARY / "deployment_paired.csv", index=False)
    write_table(
        contrasts[contrasts.contrast.isin([f"ref-{b}|model" for b in BASELINES] + [f"ref-{b}|gated" for b in BASELINES])]
        .assign(Contrast=lambda t: t.contrast.str.replace("ref-", "SG - ", regex=False).str.replace("|", ", ", regex=False))
        [["dataset", "Contrast", "mean", "low", "high", "seeds"]]
        .rename(columns={"dataset": "Dataset", "mean": "Mean", "low": "95% low", "high": "95% high", "seeds": "Seeds"}),
        "contrasts",
    )
    figure_scenarios(frame)
    figure_policy(frame)
    figure_recipe(recipe)
    figure_ablation(ablation)
    figure_freshness(runs)
    missing = status(frame)
    note = (
        "" if not missing
        else r"\pending{Partial results: fewer than five seeds for " + str(len(missing)) + " dataset--model pairs; rerun scripts/build\\_deployment\\_report.py.}"
    )
    (PAPER / "generated_dep_status.tex").write_text(note + "\n", encoding="utf-8")
    print(json.dumps({"incomplete": missing}, indent=0))


if __name__ == "__main__":
    main()
