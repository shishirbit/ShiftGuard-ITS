"""Resumable natural-missingness evaluation over every completed training run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from train import MODEL_CHOICES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="METR-LA")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for metrics in sorted((args.artifact_root / args.dataset).glob("*/seed_*/metrics.json")):
        run_dir = metrics.parent
        model, seed = run_dir.parent.name, run_dir.name.removeprefix("seed_")
        if model not in MODEL_CHOICES:
            continue
        if (run_dir / "natural_missingness.json").exists() and not args.force:
            continue
        command = [
            sys.executable, "scripts/evaluate_natural_missingness.py",
            "--dataset", args.dataset, "--model", model, "--seed", seed,
            "--artifact-root", str(args.artifact_root),
        ]
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
