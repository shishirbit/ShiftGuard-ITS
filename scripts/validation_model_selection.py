"""Select the reference ShiftGuard variant on validation data only.

Scores each variant on the validation split, clean and under the four outage
families at 30% severity (masks drawn with a dedicated validation seed, never
the saved test masks), so the choice of reference model never uses test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from evaluate_stress import infer
from train import build_model, load_split

from shiftguard.stress import burst_mask, hub_mask, independent_mask, region_mask

VARIANTS = (
    "shiftguard",
    "shiftguard_noreliability",
    "shiftguard_staticgraph",
    "shiftguard_adaptivegraph",
    "shiftguard_norouter",
    "shiftguard_noaugmentation",
    "shiftguard_independentaugmentation",
)
VALIDATION_MASK_SEED = 7_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["METR-LA", "PEMS-BAY"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 12, 42, 123, 2026])
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--output", default="validation_model_selection.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for dataset in args.datasets:
        root = args.data_root / dataset
        val_x, val_y = load_split(root, "val")
        adjacency = np.load(root / "correlation_adjacency.npy")
        with np.load(root / "scaler.npz") as archive:
            mean, std = archive["mean"], archive["std"]
        shape = (*val_x.shape[:-1], 1)
        rng = np.random.default_rng(VALIDATION_MASK_SEED)
        masks = {
            "clean": np.ones(shape, dtype=bool),
            "independent": independent_mask(shape, args.rate, rng),
            "burst": burst_mask(shape, args.rate, rng),
            "region": region_mask(shape, args.rate, adjacency, rng),
            "hub": hub_mask(shape, args.rate, adjacency),
        }
        for variant in args.variants:
            for seed in args.seeds:
                checkpoint = args.artifact_root / dataset / variant / f"seed_{seed}" / "best.pt"
                if not checkpoint.exists():
                    continue
                model = build_model(variant, val_x.shape[2], torch.from_numpy(adjacency)).to(device)
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                row = {"dataset": dataset, "model": variant, "seed": seed}
                for family, observed in masks.items():
                    point, _, target = infer(model, val_x, val_y, observed, device, 64)
                    row[family] = float(np.nanmean(np.abs((point - target) * std)))
                row["stress_mean"] = float(np.mean([row[f] for f in masks if f != "clean"]))
                rows.append(row)
                print(json.dumps(row), flush=True)
    destination = args.artifact_root / "summary"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / args.output).write_text(
        json.dumps({"rate": args.rate, "mask_seed": VALIDATION_MASK_SEED, "runs": rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
