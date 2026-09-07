from __future__ import annotations

import argparse
from pathlib import Path

from shiftguard.data import save_processed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["METR-LA", "PEMS-BAY"])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    save_processed(
        args.raw_dir / f"{args.dataset}.csv",
        args.output_root / args.dataset,
    )


if __name__ == "__main__":
    main()

