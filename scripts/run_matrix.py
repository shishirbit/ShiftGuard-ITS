from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from train import MODEL_CHOICES

from shiftguard.registry import REFERENCE_MODEL

DATASETS = ("METR-LA", "PEMS-BAY")
MODELS = MODEL_CHOICES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--skip-calibration-ablations", action="store_true")
    parser.add_argument("--stress-rates", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--microbatch-size", type=int, default=0)
    args = parser.parse_args()
    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                metrics = args.output_root / dataset / model / f"seed_{seed}" / "metrics.json"
                if metrics.exists() and not args.force:
                    print(f"skip {dataset}/{model}/seed_{seed}")
                else:
                    command = [
                        sys.executable,
                        "scripts/train.py",
                        "--dataset",
                        dataset,
                        "--model",
                        model,
                        "--seed",
                        str(seed),
                        "--epochs",
                        str(args.epochs),
                        "--patience",
                        str(args.patience),
                        "--batch-size",
                        str(args.batch_size),
                        "--output-root",
                        str(args.output_root),
                    ]
                    if args.amp:
                        command.append("--amp")
                    if args.microbatch_size:
                        command.extend(["--microbatch-size", str(args.microbatch_size)])
                    if args.resume and not args.force:
                        command.append("--resume")
                    print(" ".join(command))
                    subprocess.run(command, check=True)
                if args.skip_stress:
                    continue
                for rate in args.stress_rates:
                    destination = args.output_root / dataset / model / f"seed_{seed}"
                    base_command = [
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
                        str(args.output_root),
                        "--batch-size",
                        str(args.batch_size),
                    ]
                    primary = destination / f"stress_rate_{rate:.1f}.json"
                    if args.force or not primary.exists():
                        subprocess.run(base_command, check=True)
                    if model == REFERENCE_MODEL and not args.skip_calibration_ablations:
                        for calibration, selection in (
                            ("group-static", "risk"),
                            ("global-prequential", "risk"),
                            ("group-prequential", "all"),
                        ):
                            suffix = f"_{calibration}_{selection}"
                            result = destination / f"stress_rate_{rate:.1f}{suffix}.json"
                            if args.force or not result.exists():
                                subprocess.run(
                                    base_command
                                    + [
                                        "--calibration",
                                        calibration,
                                        "--selection",
                                        selection,
                                    ],
                                    check=True,
                                )


if __name__ == "__main__":
    main()
