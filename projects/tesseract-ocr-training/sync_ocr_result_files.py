#!/usr/bin/env python3
"""Copy core OCR result files without copying prediction text trees."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CORE_FILES = (
    "summary.csv",
    "summary.txt",
    "per_image.csv",
    "environment.txt",
    "status.txt",
    "status.json",
    "stats.txt",
    "run.log",
    "kurdish_char_errors.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync core OCR result files into another result folder.")
    parser.add_argument("--src", required=True, type=Path, help="Source result folder.")
    parser.add_argument("--dst", required=True, type=Path, help="Destination result folder.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.src.exists() or not args.src.is_dir():
        raise SystemExit(f"missing source result folder: {args.src}")

    args.dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []
    for filename in CORE_FILES:
        src_file = args.src / filename
        if not src_file.exists():
            missing.append(filename)
            continue
        shutil.copy2(src_file, args.dst / filename)
        copied.append(filename)

    print(f"Copied {len(copied)} core file(s) from {args.src} to {args.dst}")
    for filename in copied:
        print(f"  copied: {filename}")
    for filename in missing:
        print(f"  missing: {filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
