#!/usr/bin/env python3
"""Evaluate PaddleOCR Arabic-script models on image/text ground-truth pairs."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluate_tesseract_dataset import (
    Counts,
    add_counts,
    edit_counts,
    fmt,
    normalize_text,
    rate_summary,
)


DEFAULT_DATASET = Path(r"E:\TRDG\new_ds_for_finetune\test_nrt_pdf_images")
DEFAULT_OUTPUT_DIR = Path("train") / "paddle_base_eval"
ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG_LANGS = {
    "ar": "ar",
    "fa": "fa",
    "ur": "ur",
}


@dataclass(frozen=True)
class PaddleConfig:
    config_id: str
    lang: str
    display_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CER/WER/recall for PaddleOCR.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-list", type=Path, default=None)
    parser.add_argument("--split-name", default="all")
    parser.add_argument("--configs", default="ar,ur", help="Comma-separated IDs: ar,fa,ur")
    parser.add_argument("--ocr-version", default="PP-OCRv5", choices=("PP-OCRv5", "PP-OCRv4", "PP-OCRv3"))
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--image-ext", default=".png")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--status-every", type=int, default=25)
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


def parse_configs(raw: str) -> list[PaddleConfig]:
    configs: list[PaddleConfig] = []
    seen: set[str] = set()
    for item in raw.split(","):
        config_id = item.strip()
        if not config_id or config_id in seen:
            continue
        if config_id not in CONFIG_LANGS:
            raise ValueError(f"Unknown Paddle config: {config_id}")
        seen.add(config_id)
        configs.append(PaddleConfig(config_id, CONFIG_LANGS[config_id], f"PaddleOCR ({config_id})"))
    if not configs:
        raise ValueError("No valid Paddle configs requested.")
    return configs


def stems_from_list(path: Path) -> set[str]:
    stems: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            stems.add(Path(line).stem)
    return stems


def find_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    allowed = stems_from_list(args.split_list) if args.split_list else None
    pairs: list[tuple[Path, Path]] = []
    for image in sorted(args.dataset.glob(f"*{args.image_ext}")):
        if allowed is not None and image.stem not in allowed:
            continue
        text = image.with_suffix(".txt")
        if text.exists():
            pairs.append((image, text))
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise FileNotFoundError("No image/text pairs found for evaluation.")
    return pairs


def box_to_xyxy(box: Any) -> tuple[float, float, float, float]:
    if hasattr(box, "tolist"):
        box = box.tolist()
    if len(box) == 4 and all(isinstance(item, (int, float)) for item in box):
        x1, y1, x2, y2 = box
        return float(x1), float(y1), float(x2), float(y2)
    xs = [float(point[0]) for point in box]
    ys = [float(point[1]) for point in box]
    return min(xs), min(ys), max(xs), max(ys)


def extract_text(result: dict[str, Any]) -> str:
    texts = [str(text).strip() for text in result.get("rec_texts", [])]
    boxes = result.get("rec_boxes")
    if boxes is None:
        boxes = result.get("rec_polys")
    if not texts:
        return ""
    if boxes is None or len(boxes) != len(texts):
        return " ".join(text for text in texts if text)

    entries = []
    heights = []
    for text, box in zip(texts, boxes):
        if not text:
            continue
        x1, y1, x2, y2 = box_to_xyxy(box)
        height = max(1.0, y2 - y1)
        heights.append(height)
        entries.append(
            {
                "text": text,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
            }
        )
    if not entries:
        return ""

    median_height = statistics.median(heights) if heights else 20.0
    line_threshold = max(10.0, median_height * 0.65)
    entries.sort(key=lambda item: item["cy"])

    lines: list[list[dict[str, Any]]] = []
    line_centers: list[float] = []
    for entry in entries:
        best_index = None
        best_distance = None
        for idx, center in enumerate(line_centers):
            distance = abs(float(entry["cy"]) - center)
            if distance <= line_threshold and (best_distance is None or distance < best_distance):
                best_index = idx
                best_distance = distance
        if best_index is None:
            lines.append([entry])
            line_centers.append(float(entry["cy"]))
        else:
            lines[best_index].append(entry)
            line_centers[best_index] = statistics.mean(float(item["cy"]) for item in lines[best_index])

    ordered_parts: list[str] = []
    for line in lines:
        line.sort(key=lambda item: item["cx"], reverse=True)
        ordered_parts.append(" ".join(str(item["text"]) for item in line if item["text"]))
    return " ".join(part for part in ordered_parts if part)


def evaluate_config(config: PaddleConfig, pairs: list[tuple[Path, Path]], args: argparse.Namespace) -> dict[str, Any]:
    import paddle
    from paddleocr import PaddleOCR

    paddle.seed(42)
    ocr = PaddleOCR(
        lang=config.lang,
        ocr_version=args.ocr_version,
        device=args.device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
        cpu_threads=1,
    )

    config_dir = args.output_dir / config.config_id
    predictions_dir = config_dir / "predictions" / args.split_name
    predictions_dir.mkdir(parents=True, exist_ok=True)
    per_file_path = config_dir / f"{args.split_name}_per_file.csv"
    summary_path = config_dir / f"{args.split_name}_summary.txt"

    total_char = Counts()
    total_word = Counts()
    rows: list[dict[str, str]] = []
    start_all = time.perf_counter()
    for index, (image, truth_path) in enumerate(pairs, start=1):
        if args.status_every > 0 and (index == 1 or index == len(pairs) or index % args.status_every == 0):
            print(f"[{config.config_id}] OCR {index}/{len(pairs)} {image.name}", flush=True)
        ref = normalize_text(truth_path.read_text(encoding="utf-8-sig"))
        start_one = time.perf_counter()
        results = ocr.predict(str(image))
        elapsed = time.perf_counter() - start_one
        hyp = normalize_text(" ".join(extract_text(item) for item in results if isinstance(item, dict)).strip())
        (predictions_dir / f"{image.stem}.hyp.txt").write_text(hyp + "\n", encoding="utf-8")

        char_counts = edit_counts(list(ref), list(hyp))
        word_counts = edit_counts(ref.split(), hyp.split())
        total_char = add_counts(total_char, char_counts)
        total_word = add_counts(total_word, word_counts)
        char_rates = rate_summary(char_counts)
        word_rates = rate_summary(word_counts)
        rows.append(
            {
                "file": image.name,
                "chars_ref": str(char_counts.ref),
                "chars_hyp": str(char_counts.hyp),
                "cer_percent": fmt(char_rates["error_rate"]),
                "char_recall_percent": fmt(char_rates["recall"]),
                "char_precision_percent": fmt(char_rates["precision"]),
                "words_ref": str(word_counts.ref),
                "words_hyp": str(word_counts.hyp),
                "wer_percent": fmt(word_rates["error_rate"]),
                "word_recall_percent": fmt(word_rates["recall"]),
                "word_precision_percent": fmt(word_rates["precision"]),
                "time_s": fmt(elapsed),
            }
        )

    with per_file_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    char_rates = rate_summary(total_char)
    word_rates = rate_summary(total_word)
    total_elapsed = time.perf_counter() - start_all
    summary = {
        "config_id": config.config_id,
        "display_name": config.display_name,
        "language": config.lang,
        "files": len(pairs),
        "cer_percent": char_rates["error_rate"],
        "wer_percent": word_rates["error_rate"],
        "char_recall_percent": char_rates["recall"],
        "char_precision_percent": char_rates["precision"],
        "char_f1_percent": char_rates["f1"],
        "word_recall_percent": word_rates["recall"],
        "word_precision_percent": word_rates["precision"],
        "word_f1_percent": word_rates["f1"],
        "seconds_total": total_elapsed,
        "seconds_per_file": total_elapsed / len(pairs),
        "summary_path": str(summary_path),
        "per_file_csv": str(per_file_path),
        "predictions_dir": str(predictions_dir),
    }
    lines = [
        f"dataset: {args.dataset}",
        f"config_id: {config.config_id}",
        f"display_name: {config.display_name}",
        f"language: {config.lang}",
        f"ocr_version: {args.ocr_version}",
        f"device: {args.device}",
        f"split: {args.split_name}",
        f"files: {len(pairs)}",
        "",
        f"CER_percent: {fmt(summary['cer_percent'])}",
        f"char_recall_percent: {fmt(summary['char_recall_percent'])}",
        f"char_precision_percent: {fmt(summary['char_precision_percent'])}",
        f"char_f1_percent: {fmt(summary['char_f1_percent'])}",
        "",
        f"WER_percent: {fmt(summary['wer_percent'])}",
        f"word_recall_percent: {fmt(summary['word_recall_percent'])}",
        f"word_precision_percent: {fmt(summary['word_precision_percent'])}",
        f"word_f1_percent: {fmt(summary['word_f1_percent'])}",
        "",
        f"seconds_total: {fmt(total_elapsed)}",
        f"seconds_per_file: {fmt(total_elapsed / len(pairs))}",
        f"per_file_csv: {per_file_path}",
        f"predictions_dir: {predictions_dir}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return summary


def main() -> int:
    args = parse_args()
    configure_cache()
    args.dataset = args.dataset.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = parse_configs(args.configs)
    pairs = find_pairs(args)

    summaries = [evaluate_config(config, pairs, args) for config in configs]
    combined_path = args.output_dir / f"{args.split_name}_summary.csv"
    with combined_path.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = [
            "config_id",
            "display_name",
            "language",
            "files",
            "cer_percent",
            "wer_percent",
            "char_recall_percent",
            "word_recall_percent",
            "seconds_per_file",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary[key] for key in fieldnames})
    print(f"combined_summary_csv: {combined_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
