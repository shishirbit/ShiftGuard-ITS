"""Stratify forecasts by the dataset's own (natural) sensor missingness.

No synthetic masks are applied. Test origins are grouped by the real missingness
pattern in their input window, and point, interval, and selective metrics are
reported per group using the same prequential calibration and risk controller
as the stress protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from evaluate_stress import infer
from train import MODEL_CHOICES, base_model_name, build_model, load_split

from shiftguard.calibration import finite_mean, prequential_calibration

NETWORK_WIDE_STEP = 0.9  # a step is a network-wide dropout if >=90% of sensors are missing


def strata(observed: np.ndarray) -> dict[str, np.ndarray]:
    """Origin-level groups from the intrinsic input mask (samples, history, nodes)."""

    missing = 1.0 - observed.mean(axis=(1, 2))
    network_wide = ((1.0 - observed.mean(axis=2)) >= NETWORK_WIDE_STEP).any(axis=1)
    partial = ~network_wide & (missing > 0)
    return {
        "all": np.ones(len(observed), dtype=bool),
        "complete": missing == 0,
        "sensor_outage_low": partial & (missing <= 0.05),
        "sensor_outage_high": partial & (missing > 0.05),
        "network_wide": network_wide,
    }


def summarize(
    selected: np.ndarray,
    target: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    accepted: np.ndarray,
    dead_input: np.ndarray,
) -> dict[str, float | int | None]:
    error = np.abs(target - point)[selected]
    finite = np.isfinite(error)
    covered = (target >= lower) & (target <= upper)
    interval_valid = np.isfinite(target) & np.isfinite(lower) & np.isfinite(upper)
    origin_error = finite_mean(np.abs(target - point), (1, 2))[selected]
    origin_accepted = accepted[:, 0, 0][selected]
    accepted_errors = origin_error[origin_accepted & np.isfinite(origin_error)]
    dead = (dead_input & np.isfinite(target))[selected]
    return {
        "origins": int(selected.sum()),
        "mae": float(error[finite].mean()) if finite.any() else None,
        # Cells whose sensor was silent for the whole input window yet has an
        # observed target: the model must forecast from neighbors alone.
        "dead_sensor_mae": float(error[dead].mean()) if dead.any() else None,
        "dead_sensor_cells": int(dead.sum()),
        "coverage": float(covered[selected][interval_valid[selected]].mean())
        if interval_valid[selected].any()
        else None,
        "retained": float(origin_accepted.mean()) if len(origin_accepted) else None,
        "failure_rate_5mph": float(np.mean(accepted_errors > 5.0)) if len(accepted_errors) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="METR-LA", choices=["METR-LA", "PEMS-BAY"])
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if base_model_name(args.model) == "staeformer":
        args.batch_size = min(args.batch_size, 24)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = args.data_root / args.dataset
    val_x, val_y = load_split(root, "val")
    test_x, test_y = load_split(root, "test")
    adjacency = torch.from_numpy(np.load(root / "correlation_adjacency.npy"))
    model = build_model(args.model, test_x.shape[2], adjacency).to(device)
    run_dir = args.artifact_root / args.dataset / args.model / f"seed_{args.seed}"
    if sum(p.numel() for p in model.parameters() if p.requires_grad):
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    with np.load(root / "scaler.npz") as archive:
        mean, std = archive["mean"], archive["std"]

    def inverse(array):
        return array * std + mean

    # No synthetic outage: the only missing inputs are the dataset's own.
    vp, vs, vt = infer(model, val_x, val_y, np.ones((*val_x.shape[:-1], 1), bool), device, args.batch_size)
    tp, ts, tt = infer(model, test_x, test_y, np.ones((*test_x.shape[:-1], 1), bool), device, args.batch_size)
    vp, vt, vs = inverse(vp), inverse(vt), vs * std
    tp, tt, ts = inverse(tp), inverse(tt), ts * std
    lower, upper, accepted, _ = prequential_calibration(vt, vp, vs, tt, tp, ts, alpha=0.1)

    observed = test_x.numpy()[..., 1]
    dead_sensor = observed.sum(axis=1) == 0  # (samples, nodes)
    dead_input = np.broadcast_to(dead_sensor[:, None, :], tt.shape)
    results = {
        name: summarize(selected, tt, tp, lower, upper, accepted, dead_input)
        for name, selected in strata(observed).items()
    }
    payload = {
        "protocol_version": "natural-missingness-v1",
        "network_wide_step_threshold": NETWORK_WIDE_STEP,
        "results": results,
    }
    (run_dir / "natural_missingness.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
