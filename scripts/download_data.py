"""Download the two benchmark datasets from the archived Zenodo record.

The script is deliberately standard-library-only so it can run before the
research environment is installed. Files are streamed, checksum verified, and
written atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path

FILES = {
    "METR-LA.csv": (
        "https://zenodo.org/api/records/5146275/files/METR-LA.csv/content",
        "d3adaa79856bf610f25558fc242c7190",
    ),
    "PEMS-BAY.csv": (
        "https://zenodo.org/api/records/5146275/files/PEMS-BAY.csv/content",
        "c8dea58987a5882e946217c22fdb8256",
    ),
    "PEMS-BAY-META.csv": (
        "https://zenodo.org/api/records/5146275/files/PEMS-BAY-META.csv/content",
        "6427ca7d8b37140535886dd66d6f0211",
    ),
    "adj_mx_bay.pkl": (
        "https://zenodo.org/api/records/5146275/files/adj_mx_bay.pkl/content",
        "55d25daceac847312b7748c59ded6f77",
    ),
}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, url: str, checksum: str, output_dir: Path) -> None:
    destination = output_dir / name
    if destination.exists() and md5(destination) == checksum:
        print(f"verified {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"downloading {name}")
    existing = temporary.stat().st_size if temporary.exists() else 0
    request = urllib.request.Request(url)
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        partial = response.status == 206 and existing > 0
        mode = "ab" if partial else "wb"
        with temporary.open(mode) as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
                handle.flush()
    actual = md5(temporary)
    if actual != checksum:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {name}: {actual} != {checksum}")
    os.replace(temporary, destination)
    print(f"verified {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, (url, checksum) in FILES.items():
        download(filename, url, checksum, args.output_dir)


if __name__ == "__main__":
    main()
