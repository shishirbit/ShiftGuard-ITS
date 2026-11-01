"""Summarize artifacts/deployment into CSV tables; only finished seeds count."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("artifacts/deployment")
SUMMARY = Path("artifacts/summary")
OUTAGES = ("trace_x1", "trace_x2", "sensor_s30", "region_s30", "hub_s15")
PRIMARY = ("native", *OUTAGES)
RESAMPLES = 10000
# Evidence gating was piloted on this run, so gating claims must not use it
# (see the 2026-11-01 entry in docs/experiment_protocol.md).
EXPLORATORY_FOR_GATING = {("METR-LA", 1)}


def confirmatory_for_gating(dataset: str, seed: int) -> bool:
    return (dataset, seed) not in EXPLORATORY_FOR_GATING


def load_runs() -> dict[tuple[str, str, int], dict]:
    runs = {}
    for path in ROOT.glob("*/*/seed_*.json"):
        dataset, model = path.parts[-3], path.parts[-2]
        runs[(dataset, model, int(path.stem[5:]))] = json.loads(path.read_text())
    return runs


def flatten(runs: dict) -> pd.DataFrame:
    rows = []
    for (dataset, model, seed), results in runs.items():
        for scenario, record in results.items():
            if scenario.startswith("_"):
                continue
            evidence = record["evidence"]
            row = {
                "dataset": dataset, "model": model, "seed": seed, "scenario": scenario,
                **{key: record[key] for key in (
                    "cells_unavailable", "mae", "rmse", "mae_seen", "mae_blind",
                    "fallback_mae", "fallback_mae_blind", "hybrid_mae",
                )},
                "gated_mae": evidence["mae"],
                "gated_rmse": evidence["rmse"],
                "gated_peak": evidence["mae_peak"],
                "gated_offpeak": evidence["mae_offpeak"],
                "model_peak": evidence["model_mae_peak"],
                "model_offpeak": evidence["model_mae_offpeak"],
                "evidence_coverage": evidence["coverage"]["coverage"],
                "evidence_width": evidence["coverage"]["mean_width"],
                "evidence_worst_regime": evidence["worst_regime_coverage"],
                "evidence_worst_nh": evidence["worst_node_horizon_coverage"],
            }
            for name, bucket in evidence["buckets"].items():
                for key, value in bucket.items():
                    row[f"{name}_{key}"] = value
            for arm, scored in record["arms"].items():
                for key in (
                    "retained", "retained_peak", "retained_offpeak", "failure_released",
                    "failure_all", "failure_share_in_peak", "policy_mae_fallback",
                    "policy_mae_oracle", "policy_mae_random", "e_aurc", "aurc",
                    "failure_auroc", "days_over_risk_limit", "certified_blocks",
                    "coverage_blind", "coverage_seen", "worst_regime_coverage",
                    "worst_node_horizon_coverage",
                ):
                    row[f"{arm}|{key}"] = scored[key]
                row[f"{arm}|coverage"] = scored["coverage"]["coverage"]
                row[f"{arm}|width"] = scored["coverage"]["mean_width"]
            rows.append(row)
    return pd.DataFrame(rows)


def crossed_bootstrap(differences: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Interval for a ``(seed, day)`` matrix, resampling seeds and days."""

    seeds, days = differences.shape
    seed_draw = rng.integers(0, seeds, (RESAMPLES, seeds))
    day_draw = rng.integers(0, days, (RESAMPLES, days))
    draws = np.empty(RESAMPLES)
    for index in range(RESAMPLES):
        draws[index] = differences[np.ix_(seed_draw[index], day_draw[index])].mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(differences.mean()), float(low), float(high)


def paired_intervals(runs: dict, frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    rows = []
    reference = "shiftguard_noreliability"
    for dataset in sorted(frame.dataset.unique()):
        seeds = {key[2]: value for key, value in runs.items() if key[:2] == (dataset, reference)}
        for scenario in OUTAGES:
            ready = [s for s, r in seeds.items() if scenario in r and confirmatory_for_gating(dataset, s)]
            if not ready:
                continue
            gate = np.array([np.array(seeds[s][scenario]["evidence"]["daily_mae"]) - np.array(seeds[s][scenario]["daily_mae"]) for s in ready])
            rows.append({"dataset": dataset, "scenario": scenario, "contrast": "gated-model", "seeds": len(ready), **dict(zip(("mean", "low", "high"), crossed_bootstrap(gate, rng)))})
        for comparator in sorted(frame[frame.dataset == dataset].model.unique()):
            if comparator == reference:
                continue
            other = {key[2]: value for key, value in runs.items() if key[:2] == (dataset, comparator)}
            shared = sorted(set(seeds) & set(other))
            shared = [s for s in shared if all(sc in seeds[s] and sc in other[s] for sc in OUTAGES)]
            if not shared:
                continue
            for field, label in ((lambda r: r["daily_mae"], "model"), (lambda r: r["evidence"]["daily_mae"], "gated")):
                members = [s for s in shared if label == "model" or confirmatory_for_gating(dataset, s)]
                if not members:
                    continue
                matrix = np.mean([
                    [np.array(field(seeds[s][sc])) - np.array(field(other[s][sc])) for s in members]
                    for sc in OUTAGES
                ], axis=0)
                rows.append({"dataset": dataset, "scenario": "mean_outage", "contrast": f"ref-{comparator}|{label}", "seeds": len(members), **dict(zip(("mean", "low", "high"), crossed_bootstrap(matrix, rng)))})
    return pd.DataFrame(rows)


def main() -> None:
    runs = load_runs()
    frame = flatten(runs)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SUMMARY / "deployment_all_runs.csv", index=False)
    numeric = frame.drop(columns=["seed"]).groupby(["dataset", "model", "scenario"]).mean(numeric_only=True)
    numeric["seeds"] = frame.groupby(["dataset", "model", "scenario"]).seed.nunique()
    numeric.reset_index().to_csv(SUMMARY / "deployment_summary.csv", index=False)
    confirmatory = frame[[confirmatory_for_gating(d, s) for d, s in zip(frame.dataset, frame.seed)]]
    gating = confirmatory.drop(columns=["seed"]).groupby(["dataset", "model", "scenario"]).mean(numeric_only=True)
    gating["seeds"] = confirmatory.groupby(["dataset", "model", "scenario"]).seed.nunique()
    gating.reset_index().to_csv(SUMMARY / "deployment_summary_gating.csv", index=False)
    paired_intervals(runs, frame).to_csv(SUMMARY / "deployment_paired.csv", index=False)
    print(frame.groupby(["dataset", "model"]).agg(seeds=("seed", "nunique"), scenarios=("scenario", "count")).to_string())


if __name__ == "__main__":
    main()
