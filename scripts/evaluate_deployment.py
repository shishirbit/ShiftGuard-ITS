"""Deployment-feasible evaluation of trained forecasters on outage timelines.

Outages persist across forecast origins, the calibrator reads only readings a
working sensor reported, and abstention is scored against a real fallback.
Checkpoints are reused; nothing here trains or selects a model.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import torch
from evaluate_stress import infer, regime_labels
from train import MODEL_CHOICES, base_model_name, build_model, load_split

from shiftguard.calibration import (
    conformal_quantile,
    finite_mean,
    interval_metrics,
    prequential_calibration,
    risk_coverage_curve,
)
from shiftguard.evidence import (
    BUCKET_NAMES,
    blend,
    evidence_buckets,
    evidence_intervals,
    fit_blend_weights,
    shadow_availability,
)
from shiftguard.outages import (
    HISTORY,
    HORIZON,
    episode_durations,
    historical_profile,
    hub_episode_timeline,
    region_episode_timeline,
    sensor_renewal_timeline,
    trace_replay_timeline,
    windows_from_timeline,
)

SCENARIOS = (
    "native",
    "trace_x1",
    "trace_x2",
    "sensor_s10",
    "sensor_s30",
    "region_s10",
    "region_s30",
    "hub_s10",
    "hub_s15",
)
PEAK_HOURS = ((6, 10), (15, 19))
TOLERANCE = 5.0
RISK_LIMIT = 0.10


def calendar(x: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Time-of-day and day-of-week per timestamp of consecutive windows."""

    first_time = int(round(float(x[0, 0, 0, 2]) * 288)) % 288
    first_day = int(round(float(x[0, 0, 0, 3]) * 6))
    absolute = first_time + np.arange(steps)
    return absolute % 288, (first_day + absolute // 288) % 7


def scenario_timeline(
    scenario: str,
    dataset: str,
    seed: int,
    steps: int,
    nodes: int,
    trace: np.ndarray,
    durations: np.ndarray,
    adjacency: np.ndarray,
) -> np.ndarray:
    if scenario == "native":
        return np.ones((steps, nodes), dtype=bool)
    key = zlib.crc32(f"{dataset}|{scenario}|{seed}".encode())
    rng = np.random.default_rng(key)
    family, level = scenario.split("_")
    if family == "trace":
        return trace_replay_timeline(steps, nodes, trace, int(level[1:]), rng)
    severity = int(level[1:]) / 100
    if family == "sensor":
        return sensor_renewal_timeline(steps, nodes, severity, durations, rng)
    if family == "region":
        return region_episode_timeline(steps, severity, durations, adjacency, rng)
    return hub_episode_timeline(steps, severity, durations, adjacency, rng)


def auroc(score: np.ndarray, positive: np.ndarray) -> float | None:
    positives, negatives = int(positive.sum()), int((~positive).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average ties so a score that is constant within a block earns no credit.
    values, inverse = np.unique(score, return_inverse=True)
    ranks = (np.bincount(inverse, ranks) / np.bincount(inverse))[inverse]
    return float((ranks[positive].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def score_arm(
    target: np.ndarray,
    point: np.ndarray,
    fallback: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    accepted_cells: np.ndarray,
    blind: np.ndarray,
    regimes: np.ndarray,
    peak: np.ndarray,
    day: np.ndarray,
) -> dict[str, object]:
    """All scores use withheld truth, including at failed sensors."""

    absolute = np.abs(target - point)
    accepted = accepted_cells[:, 0, 0]
    origin_error = finite_mean(absolute, (1, 2))
    fallback_error = finite_mean(np.abs(target - fallback), (1, 2))
    width = upper - lower
    uncertainty = finite_mean(width, (1, 2))
    valid = np.isfinite(origin_error) & np.isfinite(uncertainty) & np.isfinite(fallback_error)
    failure = origin_error > TOLERANCE
    coverage_share = float(accepted[valid].mean())

    policy_error = np.where(accepted, origin_error, fallback_error)
    released_failure = failure[valid & accepted]
    _, _, aurc = risk_coverage_curve(origin_error[valid], uncertainty[valid])
    _, _, oracle_aurc = risk_coverage_curve(origin_error[valid], origin_error[valid])

    finite = np.isfinite(target) & np.isfinite(lower) & np.isfinite(upper)
    covered = finite & (target >= lower) & (target <= upper)
    regime_coverage = {
        str(name): float(covered[regimes == name].sum() / max(1, finite[regimes == name].sum()))
        for name in np.unique(regimes)
    }
    group_count = finite.sum(axis=0)
    node_horizon = np.divide(
        covered.sum(axis=0), group_count,
        out=np.full(group_count.shape, np.nan), where=group_count > 0,
    )
    blind_cells = finite & blind
    seen_cells = finite & ~blind

    order = np.argsort(origin_error[valid], kind="stable")
    kept = max(1, int(round(coverage_share * valid.sum())))
    oracle_keep = np.zeros(valid.sum(), dtype=bool)
    oracle_keep[order[:kept]] = True
    oracle_policy = np.where(oracle_keep, origin_error[valid], fallback_error[valid]).mean()

    days = np.unique(day[valid])
    daily_failure = []
    for value in days:
        selected = valid & accepted & (day == value)
        if selected.sum() >= 12:
            daily_failure.append(float(failure[selected].mean()))
    return {
        "coverage": interval_metrics(target, lower, upper, alpha=0.1),
        "coverage_blind": float(covered[blind_cells].mean()) if blind_cells.any() else None,
        "coverage_seen": float(covered[seen_cells].mean()) if seen_cells.any() else None,
        "worst_regime_coverage": min(regime_coverage.values()),
        "regime_coverage": regime_coverage,
        "worst_node_horizon_coverage": float(np.nanmin(node_horizon)),
        "retained": coverage_share,
        "retained_peak": float(accepted[valid & peak].mean()) if (valid & peak).any() else None,
        "retained_offpeak": float(accepted[valid & ~peak].mean()) if (valid & ~peak).any() else None,
        "failure_released": float(released_failure.mean()) if len(released_failure) else None,
        "failure_all": float(failure[valid].mean()),
        "failure_share_in_peak": float(peak[valid & failure].mean()) if (valid & failure).any() else None,
        "policy_mae_fallback": float(policy_error[valid].mean()),
        "policy_mae_oracle": float(oracle_policy),
        "policy_mae_random": float(
            coverage_share * origin_error[valid].mean()
            + (1 - coverage_share) * fallback_error[valid].mean()
        ),
        "aurc": aurc,
        "e_aurc": aurc - oracle_aurc,
        "failure_auroc": auroc(uncertainty[valid], failure[valid]),
        "days_scored": len(daily_failure),
        "days_over_risk_limit": float(np.mean(np.asarray(daily_failure) > RISK_LIMIT)) if daily_failure else None,
    }


def score_evidence(
    scenario: str,
    seed: int,
    context: dict,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    gate: dict,
    feedback: np.ndarray,
    target: np.ndarray,
    point: np.ndarray,
    scale: np.ndarray,
    input_up: np.ndarray,
) -> dict[str, object]:
    """Blend by freshness and calibrate with shadow-blinded working sensors."""

    native_up = context["x_np"][..., 1] > 0
    bucket = evidence_buckets(input_up & native_up)
    gated = blend(point, context["profile"], bucket, gate["weights"])
    rng = np.random.default_rng(zlib.crc32(f"shadow|{context['dataset']}|{scenario}|{seed}".encode()))
    shadow_up, hidden = shadow_availability(input_up, rng)
    shadow_point, shadow_scale, _ = infer(
        model, context["x"], context["y"], shadow_up[..., None], device, batch_size
    )
    shadow_point = context["inverse"](shadow_point)
    shadow_scale = shadow_scale * context["std"]
    shadow_bucket = evidence_buckets(shadow_up & native_up)
    shadow_gated = blend(shadow_point, context["profile"], shadow_bucket, gate["weights"])
    shadow_score = np.abs(feedback - shadow_gated) / np.maximum(shadow_scale, 1e-4)
    shadow_score = np.where(hidden[:, None, :], shadow_score, np.nan)
    live = {
        "score": (np.abs(feedback - gated) / np.maximum(scale, 1e-4)).astype(np.float32),
        "bucket": bucket,
        "shadow_score": shadow_score.astype(np.float32),
        "shadow_bucket": shadow_bucket,
    }
    lower, upper = evidence_intervals(gate["prior"], live, gated, scale)
    finite = np.isfinite(target)
    covered = finite & (target >= lower) & (target <= upper)
    cell_bucket = np.repeat(bucket[:, None, :], point.shape[1], axis=1)
    absolute = np.abs(target - gated)
    per_bucket = {}
    for level, name in enumerate(BUCKET_NAMES):
        chosen = finite & (cell_bucket == level)
        per_bucket[name] = {
            "share": float(chosen.sum() / finite.sum()),
            "mae_model": float(np.abs(target - point)[chosen].mean()) if chosen.any() else None,
            "mae_gated": float(absolute[chosen].mean()) if chosen.any() else None,
            "coverage": float(covered[chosen].mean()) if chosen.any() else None,
            "width": float((upper - lower)[chosen].mean()) if chosen.any() else None,
        }
    peak_cells = np.repeat(context["peak"], np.prod(point.shape[1:])).reshape(point.shape)
    regime_coverage = {
        str(name): float(covered[context["regimes"] == name].sum() / max(1, finite[context["regimes"] == name].sum()))
        for name in np.unique(context["regimes"])
    }
    group_count = finite.sum(axis=0)
    node_horizon = np.divide(
        covered.sum(axis=0), group_count,
        out=np.full(group_count.shape, np.nan), where=group_count > 0,
    )
    return {
        "mae": float(absolute[finite].mean()),
        "rmse": float(np.sqrt((absolute[finite] ** 2).mean())),
        "mae_peak": float(absolute[finite & peak_cells].mean()),
        "mae_offpeak": float(absolute[finite & ~peak_cells].mean()),
        "model_mae_peak": float(np.abs(target - point)[finite & peak_cells].mean()),
        "model_mae_offpeak": float(np.abs(target - point)[finite & ~peak_cells].mean()),
        "coverage": interval_metrics(target, lower, upper, alpha=0.1),
        "buckets": per_bucket,
        "worst_regime_coverage": min(regime_coverage.values()),
        "regime_coverage": regime_coverage,
        "worst_node_horizon_coverage": float(np.nanmin(node_horizon)),
        "daily_mae": [
            float(np.nanmean(absolute[context["day"] == value]))
            for value in np.unique(context["day"])
        ],
    }


def fit_gate(
    seed: int,
    context: dict,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    calibration: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict:
    """Fit weights and the calibration prior on validation data only.

    Validation readings are archived, so hiding sensors there leaves truth to
    score against. These outage draws use their own seeds and never test data.
    """

    val_x, val_y = context["val_x"], context["val_y"]
    val_np = val_x.numpy()
    origins, _, nodes, _ = val_np.shape
    steps = origins + HISTORY + HORIZON - 1
    time_index, day_index = calendar(val_np, steps)
    target_steps = np.arange(origins)[:, None] + HISTORY + np.arange(HORIZON)[None]
    profile = context["profile_table"][day_index[target_steps], time_index[target_steps]]
    native_up = val_np[..., 1] > 0
    cal_target, cal_point, cal_scale = calibration
    rng = np.random.default_rng(zlib.crc32(f"validation-gate|{context['dataset']}|{seed}".encode()))
    timelines = [
        sensor_renewal_timeline(steps, nodes, 0.2, context["durations"], rng),
        region_episode_timeline(steps, 0.2, context["durations"], context["adjacency"], rng),
    ]
    targets, points, profiles, buckets, scales, availability = [], [], [], [], [], []
    for timeline in timelines:
        input_up, _ = windows_from_timeline(timeline)
        point, scale, target = infer(model, val_x, val_y, input_up[..., None], device, batch_size)
        targets.append(context["inverse"](target))
        points.append(context["inverse"](point))
        scales.append(scale * context["std"])
        profiles.append(profile)
        buckets.append(evidence_buckets(input_up & native_up))
        availability.append(input_up)
    weights = fit_blend_weights(
        np.concatenate(targets), np.concatenate(points), np.concatenate(profiles), np.concatenate(buckets)
    )
    clean_bucket = evidence_buckets(native_up)
    clean_gated = blend(cal_point, profile, clean_bucket, weights)
    # The prior alternates between timelines so that its recent window, which
    # survives the rolling cut, still holds silent-sensor residuals.
    outage_score = np.where(
        (np.arange(origins) % 2 == 0)[:, None, None],
        np.abs(targets[0] - blend(points[0], profile, buckets[0], weights)) / np.maximum(scales[0], 1e-4),
        np.abs(targets[1] - blend(points[1], profile, buckets[1], weights)) / np.maximum(scales[1], 1e-4),
    )
    outage_bucket = np.where((np.arange(origins) % 2 == 0)[:, None], buckets[0], buckets[1])
    prior = {
        "score": (np.abs(cal_target - clean_gated) / np.maximum(cal_scale, 1e-4)).astype(np.float32),
        "bucket": clean_bucket,
        "shadow_score": outage_score.astype(np.float32),
        "shadow_bucket": outage_bucket,
    }
    return {"weights": weights, "prior": prior}


def run_scenario(
    scenario: str,
    arms: list[str],
    context: dict,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    calibration: tuple[np.ndarray, np.ndarray, np.ndarray],
    seed: int,
    gate: dict,
) -> dict[str, object]:
    timeline = scenario_timeline(
        scenario, context["dataset"], seed, context["steps"], context["nodes"],
        context["trace"], context["durations"], context["adjacency"],
    )
    input_up, target_up = windows_from_timeline(timeline)
    point, scale, target = infer(
        model, context["x"], context["y"], input_up[..., None], device, batch_size
    )
    point, target, scale = context["inverse"](point), context["inverse"](target), scale * context["std"]
    feedback = np.where(target_up, target, np.nan)
    # Blind cells: no reading from this sensor anywhere in the input window.
    native_up = context["x_np"][..., 1] > 0
    blind_node = ~(input_up & native_up).any(axis=1)
    blind = np.repeat(blind_node[:, None, :], HORIZON, axis=1)

    fallback = context["profile"]
    valid = np.isfinite(target)
    absolute = np.abs(target - point)
    fallback_absolute = np.abs(target - fallback)
    hybrid = np.where(blind, fallback, point)
    seen = valid & ~blind
    blind_valid = valid & blind
    record: dict[str, object] = {
        "cells_unavailable": float(1 - input_up.mean()),
        "origin_share_with_blind_sensor": float(blind_node.any(axis=1).mean()),
        "mae": float(absolute[valid].mean()),
        "rmse": float(np.sqrt((absolute[valid] ** 2).mean())),
        "mae_seen": float(absolute[seen].mean()),
        "mae_blind": float(absolute[blind_valid].mean()) if blind_valid.any() else None,
        "fallback_mae": float(fallback_absolute[valid].mean()),
        "fallback_mae_blind": float(fallback_absolute[blind_valid].mean()) if blind_valid.any() else None,
        "hybrid_mae": float(np.abs(target - hybrid)[valid].mean()),
        "daily_mae": [
            float(np.nanmean(absolute[context["day"] == value]))
            for value in np.unique(context["day"])
        ],
        "arms": {},
        "evidence": score_evidence(
            scenario, seed, context, model, device, batch_size, gate,
            feedback, target, point, scale, input_up,
        ),
    }
    for arm in arms:
        lower, upper, accepted, rules = prequential_calibration(
            *calibration,
            target,
            point,
            scale,
            alpha=0.1,
            feedback_target=None if arm == "oracle_feedback" else feedback,
            certificate_stride=HORIZON if arm == "feasible_thinned" else 1,
        )
        scored = score_arm(
            target, point, fallback, lower, upper, accepted, blind,
            context["regimes"], context["peak"], context["day"],
        )
        scored["certified_blocks"] = float(np.mean([np.isfinite(r.uncertainty_threshold) for r in rules]))
        record["arms"][arm] = scored
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["METR-LA", "PEMS-BAY"])
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 12, 42, 123, 2026])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--arms", nargs="+", default=["oracle_feedback", "feasible", "feasible_thinned"])
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with np.load(args.data_root / "METR-LA" / "train.npz") as archive:
        source_trace = archive["x"][:, 0, :, 1] > 0
    durations = episode_durations(source_trace)

    for dataset in args.datasets:
        root = args.data_root / dataset
        with np.load(root / "scaler.npz") as archive:
            mean, std = archive["mean"], archive["std"]
        inverse = lambda array, mean=mean, std=std: array * std + mean
        val_x, val_y = load_split(root, "val")
        test_x, test_y = load_split(root, "test")
        x_np, y_np = test_x.numpy(), test_y.numpy()
        adjacency = np.load(root / "correlation_adjacency.npy")
        origins, _, nodes, _ = x_np.shape
        steps = origins + HISTORY + HORIZON - 1
        time_index, day_index = calendar(x_np, steps)

        with np.load(root / "train.npz") as archive:
            train_x = archive["x"]
        train_time, train_day = calendar(train_x, len(train_x))
        profile_table = historical_profile(
            inverse(train_x[:, 0, :, 0]), train_x[:, 0, :, 1] > 0, train_time, train_day
        )
        del train_x
        target_steps = np.arange(origins)[:, None] + HISTORY + np.arange(HORIZON)[None]
        profile = profile_table[day_index[target_steps], time_index[target_steps]]

        origin_step = np.arange(origins) + HISTORY - 1
        hour = time_index[origin_step] / 12
        weekday = day_index[origin_step] < 5
        peak = weekday & np.logical_or.reduce([(hour >= a) & (hour < b) for a, b in PEAK_HOURS])
        day = (origin_step + int(round(float(x_np[0, 0, 0, 2]) * 288))) // 288
        raw_y = inverse(y_np[..., 0])
        raw_y = np.where(y_np[..., 1] > 0, raw_y, np.nan)
        regimes = regime_labels(inverse(x_np[..., 0])[..., None], raw_y[..., None])
        context = {
            "dataset": dataset, "steps": steps, "nodes": nodes, "trace": source_trace,
            "durations": durations, "adjacency": adjacency, "inverse": inverse, "std": std,
            "x": test_x, "y": test_y, "x_np": x_np, "profile": profile,
            "regimes": regimes, "peak": peak, "day": day,
            "val_x": val_x, "val_y": val_y, "profile_table": profile_table,
        }
        for name in args.models:
            for seed in args.seeds:
                run = args.artifact_root / dataset / name / f"seed_{seed}"
                checkpoint = run / "best.pt"
                if not checkpoint.exists() or not (run / "metrics.json").exists():
                    print(json.dumps({"skip_untrained": [dataset, name, seed]}), flush=True)
                    continue
                destination = args.artifact_root / "deployment" / dataset / name / f"seed_{seed}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                results = json.loads(destination.read_text()) if destination.exists() else {}
                missing = [s for s in args.scenarios if s not in results]
                if not missing:
                    continue
                model = build_model(name, nodes, torch.from_numpy(adjacency)).to(device)
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                batch_size = 16 if base_model_name(name) == "staeformer" else 64
                clean = np.ones((*val_x.shape[:-1], 1), dtype=bool)
                vp, vs, vt = infer(model, val_x, val_y, clean, device, batch_size)
                calibration = (inverse(vt), inverse(vp), vs * std)
                gate = fit_gate(seed, context, model, device, batch_size, calibration)
                results["_gate_weights"] = gate["weights"].tolist()
                for scenario in missing:
                    results[scenario] = run_scenario(
                        scenario, args.arms, context, model, device, batch_size, calibration, seed, gate
                    )
                    temporary = destination.with_suffix(".tmp")
                    temporary.write_text(json.dumps(results), encoding="utf-8")
                    temporary.replace(destination)
                    print(json.dumps({"done": [dataset, name, seed, scenario]}), flush=True)
                del model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
