#!/usr/bin/env python3
"""OCR Surya/CRAFT text-line annotations with PaddleOCR recognition."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = Path(r"E:\Antigravity_Code\CRAFT\surya_kurdish\images")
DEFAULT_ANNOTATIONS = Path(r"E:\Antigravity_Code\CRAFT\surya_kurdish\annotations")
DEFAULT_MODEL_DIR = (
    ROOT_DIR / "train" / "paddleocr_rec_arabic_v3_line_w1024_e1" / "inference"
)
DEFAULT_OUTPUT = (
    ROOT_DIR
    / "train"
    / "paddleocr_rec_arabic_v3_line_w1024_e1"
    / "surya_kurdish_ocr_results.txt"
)

ARABIC_LANGS = {"ar", "fa", "ug", "ur", "ps", "ku", "sd", "bal"}


@dataclass(frozen=True)
class LineCrop:
    index: int
    bbox: tuple[int, int, int, int]
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR recognition on annotated text-line crops and write one txt file."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Exported Paddle inference model directory. Use an empty string for a base model.",
    )
    parser.add_argument(
        "--model-name",
        default="arabic_PP-OCRv3_mobile_rec",
        help="Paddle model name. For base PP-OCRv5 Arabic/Urdu use arabic_PP-OCRv5_mobile_rec.",
    )
    parser.add_argument("--lang", default="ar")
    parser.add_argument("--ocr-version", default="PP-OCRv5", choices=("PP-OCRv3", "PP-OCRv5"))
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--input-shape", nargs=3, type=int, default=(3, 48, 1024))
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Maximum annotation files to process.")
    parser.add_argument("--keep-crops", action="store_true")
    return parser.parse_args()


def configure_cache() -> None:
    cache_home = ROOT_DIR / ".cache" / "home"
    paddlex_cache = ROOT_DIR / ".cache" / "paddlex"
    cache_home.mkdir(parents=True, exist_ok=True)
    paddlex_cache.mkdir(parents=True, exist_ok=True)
    os.environ["USERPROFILE"] = str(cache_home)
    os.environ["HOME"] = str(cache_home)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(paddlex_cache)
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def normalize_optional_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    return Path(raw)


def default_model_name(lang: str, ocr_version: str) -> str:
    if lang not in ARABIC_LANGS:
        raise ValueError(f"Only Arabic-script base model names are mapped here, got: {lang}")
    if ocr_version == "PP-OCRv5":
        return "arabic_PP-OCRv5_mobile_rec"
    return "arabic_PP-OCRv3_mobile_rec"


def build_recognizer(args: argparse.Namespace) -> Any:
    configure_cache()
    from paddleocr import TextRecognition

    model_dir = normalize_optional_path(args.model_dir)
    model_name = args.model_name.strip() if args.model_name else ""
    if not model_name:
        model_name = default_model_name(args.lang, args.ocr_version)
    kwargs: dict[str, Any] = {
        "model_name": model_name,
        "device": args.device,
        "input_shape": tuple(args.input_shape),
    }
    if model_dir is not None:
        kwargs["model_dir"] = str(model_dir.resolve())
    return TextRecognition(**kwargs)


def find_image(images_dir: Path, filename: str) -> Path:
    candidate = images_dir / filename
    if candidate.exists():
        return candidate
    stem = Path(filename).stem
    for suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for annotation entry: {filename}")


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


def sorted_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(lines, key=lambda item: (round(float(item["bbox"][1]) / 20), -float(item["bbox"][0])))


def crop_annotation(
    args: argparse.Namespace, annotation_path: Path, temp_dir: Path
) -> tuple[Path, list[LineCrop]]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_path = find_image(args.images, annotation.get("file", annotation_path.with_suffix(".jpg").name))
    lines = sorted_lines(annotation.get("text_lines", []))

    crops: list[LineCrop] = []
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        for index, line in enumerate(lines, start=1):
            bbox = clamp_bbox(line["bbox"], image.width, image.height, args.padding)
            crop = image.crop(bbox)
            crop_path = temp_dir / f"{annotation_path.stem}_{index:04d}.png"
            crop.save(crop_path)
            crops.append(LineCrop(index=index, bbox=bbox, path=crop_path))
    return image_path, crops


def result_to_text_and_score(result: Any) -> tuple[str, str]:
    data = result
    if not isinstance(data, dict):
        if hasattr(data, "json"):
            try:
                data = data.json
            except TypeError:
                data = data.json()
        elif hasattr(data, "to_dict"):
            data = data.to_dict()

    if isinstance(data, dict):
        text = (
            data.get("rec_text")
            or data.get("text")
            or data.get("label")
            or data.get("rec_texts")
            or ""
        )
        if isinstance(text, list):
            text = " ".join(str(item) for item in text if item)
        score = data.get("rec_score") or data.get("score") or data.get("scores") or ""
        if isinstance(score, list):
            score = ",".join(f"{float(item):.4f}" for item in score if isinstance(item, (int, float)))
        elif isinstance(score, (int, float)):
            score = f"{float(score):.4f}"
        return str(text).strip(), str(score).strip()
    return str(result).strip(), ""


def recognize_crops(recognizer: Any, crops: list[LineCrop], batch_size: int) -> list[tuple[LineCrop, str, str]]:
    rows: list[tuple[LineCrop, str, str]] = []
    for start in range(0, len(crops), max(1, batch_size)):
        batch = crops[start : start + max(1, batch_size)]
        results = recognizer.predict([str(crop.path) for crop in batch])
        for crop, result in zip(batch, results):
            text, score = result_to_text_and_score(result)
            rows.append((crop, text, score))
    return rows


def main() -> int:
    args = parse_args()
    args.images = args.images.resolve()
    args.annotations = args.annotations.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    annotation_files = sorted(args.annotations.glob("*.json"))
    if args.limit > 0:
        annotation_files = annotation_files[: args.limit]
    if not annotation_files:
        raise FileNotFoundError(f"No JSON annotation files found in {args.annotations}")

    temp_dir = args.output.parent / "_surya_paddle_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    recognizer = build_recognizer(args)

    try:
        with args.output.open("w", encoding="utf-8", newline="\n") as out:
            out.write(f"images: {args.images}\n")
            out.write(f"annotations: {args.annotations}\n")
            out.write(f"model_dir: {normalize_optional_path(args.model_dir)}\n")
            out.write(f"model_name: {args.model_name}\n")
            out.write(f"lang: {args.lang}\n")
            out.write(f"ocr_version: {args.ocr_version}\n")
            out.write(f"device: {args.device}\n")
            out.write(f"input_shape: {tuple(args.input_shape)}\n")
            out.write("=" * 80 + "\n\n")

            for page_index, annotation_path in enumerate(annotation_files, start=1):
                print(f"OCR {page_index}/{len(annotation_files)} {annotation_path.name}", flush=True)
                out.write(f"FILE: {annotation_path.stem}\n")
                out.write("-" * 80 + "\n")
                try:
                    image_path, crops = crop_annotation(args, annotation_path, temp_dir)
                    out.write(f"image: {image_path.name}\n")
                    out.write(f"lines: {len(crops)}\n")
                    rows = recognize_crops(recognizer, crops, args.batch_size)
                    page_text: list[str] = []
                    for crop, text, score in rows:
                        x1, y1, x2, y2 = crop.bbox
                        score_part = score if score else "-"
                        out.write(f"{crop.index:04d}\t[{x1},{y1},{x2},{y2}]\t{score_part}\t{text}\n")
                        if text:
                            page_text.append(text)
                    out.write("\nPAGE_TEXT:\n")
                    out.write("\n".join(page_text) if page_text else "[no OCR text]")
                    out.write("\n\n")
                except Exception as exc:
                    out.write(f"[ERROR] {exc}\n\n")
    finally:
        if hasattr(recognizer, "close"):
            recognizer.close()
        if not args.keep_crops:
            shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
