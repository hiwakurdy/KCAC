#!/usr/bin/env python3
"""OCR images using Surya/CRAFT text-line annotations and one Tesseract model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


DEFAULT_IMAGES = Path(r"E:\Antigravity_Code\CRAFT\surya_kurdish\images")
DEFAULT_ANNOTATIONS = Path(r"E:\Antigravity_Code\CRAFT\surya_kurdish\annotations")
DEFAULT_TESSDATA = Path("train") / "urd" / "output"
DEFAULT_OUTPUT = Path("train") / "urd" / "surya_kurdish_ocr_results.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tesseract OCR on annotated text-line crops and write one txt file."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--tessdata-dir", type=Path, default=DEFAULT_TESSDATA)
    parser.add_argument("--lang", default="urd")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument("--psm", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--keep-empty", action="store_true")
    return parser.parse_args()


def find_image(images_dir: Path, filename: str) -> Path:
    candidate = images_dir / filename
    if candidate.exists():
        return candidate
    stem = Path(filename).stem
    for suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for annotation file entry: {filename}")


def clamp_bbox(
    bbox: list[float], image_width: int, image_height: int, padding: int
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    left = max(0, int(round(x)) - padding)
    top = max(0, int(round(y)) - padding)
    right = min(image_width, int(round(x + width)) + padding)
    bottom = min(image_height, int(round(y + height)) + padding)
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid bbox after clamping: {bbox}")
    return left, top, right, bottom


def sorted_lines(lines: list[dict]) -> list[dict]:
    # For Arabic-script pages, sort top-to-bottom and right-to-left within a row.
    return sorted(lines, key=lambda item: (round(float(item["bbox"][1]) / 20), -float(item["bbox"][0])))


def run_tesseract(args: argparse.Namespace, crop_path: Path) -> str:
    proc = subprocess.run(
        [
            args.tesseract,
            str(crop_path),
            "stdout",
            "--psm",
            str(args.psm),
            "--dpi",
            str(args.dpi),
            "-l",
            args.lang,
            "--tessdata-dir",
            str(args.tessdata_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"Tesseract failed on {crop_path}")
    return " ".join(proc.stdout.split())


def ocr_annotation(args: argparse.Namespace, annotation_path: Path, temp_dir: Path) -> list[str]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_path = find_image(args.images, annotation.get("file", annotation_path.with_suffix(".jpg").name))
    lines = sorted_lines(annotation.get("text_lines", []))

    results: list[str] = []
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        for index, line in enumerate(lines, start=1):
            bbox = clamp_bbox(line["bbox"], image.width, image.height, args.padding)
            crop = image.crop(bbox)
            crop_path = temp_dir / f"{annotation_path.stem}_{index:04d}.png"
            crop.save(crop_path)
            try:
                text = run_tesseract(args, crop_path)
            finally:
                crop_path.unlink(missing_ok=True)
            if text or args.keep_empty:
                x1, y1, x2, y2 = bbox
                results.append(f"{index:04d}\t[{x1},{y1},{x2},{y2}]\t{text}")
    return results


def main() -> int:
    args = parse_args()
    args.images = args.images.resolve()
    args.annotations = args.annotations.resolve()
    args.tessdata_dir = args.tessdata_dir.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    annotation_files = sorted(args.annotations.glob("*.json"))
    if not annotation_files:
        raise FileNotFoundError(f"No JSON annotation files found in {args.annotations}")

    temp_dir = args.output.parent / "_surya_tess_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        out.write(f"images: {args.images}\n")
        out.write(f"annotations: {args.annotations}\n")
        out.write(f"tessdata: {args.tessdata_dir}\n")
        out.write(f"language: {args.lang}\n")
        out.write(f"psm: {args.psm}\n")
        out.write("=" * 80 + "\n\n")

        for page_index, annotation_path in enumerate(annotation_files, start=1):
            print(f"OCR {page_index}/{len(annotation_files)} {annotation_path.name}", flush=True)
            out.write(f"FILE: {annotation_path.stem}\n")
            out.write("-" * 80 + "\n")
            try:
                results = ocr_annotation(args, annotation_path, temp_dir)
                if results:
                    out.write("\n".join(results))
                    out.write("\n")
                else:
                    out.write("[no OCR text]\n")
            except Exception as exc:
                out.write(f"[ERROR] {exc}\n")
            out.write("\n")

    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
