from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from shiftguard.registry import REFERENCE_MODEL


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("group_coverage_version") == "node-horizon-regime-v1"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    for dataset in ("METR-LA", "PEMS-BAY"):
        for model in (REFERENCE_MODEL, "staeformer"):
            for seed in (1, 12, 42, 123, 2026):
                for rate in (0.1, 0.3, 0.5):
                    result = Path(
                        f"artifacts/{dataset}/{model}/seed_{seed}/stress_rate_{rate:.1f}.json"
                    )
                    if complete(result):
                        print(f"skip {dataset}/{model}/{seed}/{rate:.1f}", flush=True)
                        continue
                    command = [
                        sys.executable, "scripts/evaluate_stress.py", "--dataset", dataset,
                        "--model", model, "--seed", str(seed), "--rate", str(rate),
                        "--artifact-root", "artifacts", "--batch-size", "64",
                    ]
                    print(" ".join(command), flush=True)
                    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
