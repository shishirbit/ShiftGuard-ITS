from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


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
    "shiftguard",
)
ABLATIONS = (
    "shiftguard_noreliability",
    "shiftguard_staticgraph",
    "shiftguard_adaptivegraph",
    "shiftguard_norouter",
    "shiftguard_noaugmentation",
    "shiftguard_independentaugmentation",
)
RISK_UNIT = "forecast_origin_mean_absolute_error"
PROTOCOL_VERSION = "origin-risk-v1"


def current_result(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            payload.get("risk_unit") == RISK_UNIT
            and payload.get("protocol_version") == PROTOCOL_VERSION
        )
    except (OSError, json.JSONDecodeError):
        return False


def evaluate(
    artifact_root: Path,
    dataset: str,
    model: str,
    seed: int,
    rate: float,
    batch_size: int,
    calibration: str = "group-prequential",
    selection: str = "risk",
) -> None:
    suffix = "" if (calibration, selection) == ("group-prequential", "risk") else (
        f"_{calibration}_{selection}"
    )
    destination = artifact_root / dataset / model / f"seed_{seed}"
    result = destination / f"stress_rate_{rate:.1f}{suffix}.json"
    if current_result(result):
        print(f"skip {dataset}/{model}/seed_{seed}/rate_{rate:.1f}{suffix}", flush=True)
        return
    command = [
        sys.executable,
        "scripts/evaluate_stress.py",
        "--dataset",
        dataset,
        "--model",
        model,
        "--seed",
        str(seed),
        "--rate",
        str(rate),
        "--artifact-root",
        str(artifact_root),
        "--batch-size",
        str(batch_size),
        "--calibration",
        calibration,
        "--selection",
        selection,
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable confirmatory stress matrix.")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--datasets", nargs="+", default=["METR-LA", "PEMS-BAY"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 12, 42, 123, 2026])
    parser.add_argument("--rates", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    parser.add_argument("--ablation-rate", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    for dataset in args.datasets:
        for model in PRIMARY_MODELS:
            for seed in args.seeds:
                for rate in args.rates:
                    evaluate(args.artifact_root, dataset, model, seed, rate, args.batch_size)

    for dataset in args.datasets:
        for model in ABLATIONS:
            for seed in args.seeds:
                evaluate(
                    args.artifact_root,
                    dataset,
                    model,
                    seed,
                    args.ablation_rate,
                    args.batch_size,
                )

    for dataset in args.datasets:
        for seed in args.seeds:
            for calibration, selection in (
                ("group-static", "risk"),
                ("global-prequential", "risk"),
                ("group-prequential", "all"),
            ):
                evaluate(
                    args.artifact_root,
                    dataset,
                    "shiftguard",
                    seed,
                    args.ablation_rate,
                    args.batch_size,
                    calibration,
                    selection,
                )


if __name__ == "__main__":
    main()
