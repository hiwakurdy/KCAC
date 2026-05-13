#!/usr/bin/env python3
"""Evaluate a Tesseract model on image/text ground-truth pairs."""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASET = Path(r"E:\TRDG\new_ds_for_finetune\test_nrt_pdf_images")
DEFAULT_TESSDATA = Path("train") / "urd" / "output"
DEFAULT_OUTPUT_DIR = Path("train") / "urd" / "metrics"


@dataclass(frozen=True)
class Counts:
    ref: int = 0
    hyp: int = 0
    hits: int = 0
    sub: int = 0
    delete: int = 0
    insert: int = 0

    @property
    def edits(self) -> int:
        return self.sub + self.delete + self.insert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CER/WER and recall for Tesseract OCR.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tessdata-dir", type=Path, default=DEFAULT_TESSDATA)
    parser.add_argument("--lang", default="urd")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-list", type=Path, default=None)
    parser.add_argument("--split-name", default="all")
    parser.add_argument("--image-ext", default=".png")
    parser.add_argument("--psm", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tesseract", default="tesseract")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def edit_counts(ref_items: list[str], hyp_items: list[str]) -> Counts:
    rows = len(ref_items) + 1
    cols = len(hyp_items) + 1
    dp = [[0] * cols for _ in range(rows)]
    back = [[""] * cols for _ in range(rows)]

    for i in range(1, rows):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, cols):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, rows):
        for j in range(1, cols):
            if ref_items[i - 1] == hyp_items[j - 1]:
                choices = [(dp[i - 1][j - 1], "M")]
            else:
                choices = [(dp[i - 1][j - 1] + 1, "S")]
            choices.extend(
                [
                    (dp[i - 1][j] + 1, "D"),
                    (dp[i][j - 1] + 1, "I"),
                ]
            )
            dp[i][j], back[i][j] = min(choices, key=lambda item: item[0])

    hits = sub = delete = insert = 0
    i = len(ref_items)
    j = len(hyp_items)
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "M":
            hits += 1
            i -= 1
            j -= 1
        elif op == "S":
            sub += 1
            i -= 1
            j -= 1
        elif op == "D":
            delete += 1
            i -= 1
        else:
            insert += 1
            j -= 1

    return Counts(
        ref=len(ref_items),
        hyp=len(hyp_items),
        hits=hits,
        sub=sub,
        delete=delete,
        insert=insert,
    )


def pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0 if numerator == 0 else 100.0
    return 100.0 * numerator / denominator


def rate_summary(counts: Counts) -> dict[str, float]:
    precision = pct(counts.hits, counts.hyp)
    recall = pct(counts.hits, counts.ref)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "error_rate": pct(counts.edits, counts.ref),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


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


def run_ocr(args: argparse.Namespace, image: Path) -> str:
    proc = subprocess.run(
        [
            args.tesseract,
            str(image),
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
        raise RuntimeError(proc.stderr.strip() or f"Tesseract failed on {image}")
    return normalize_text(proc.stdout)


def add_counts(left: Counts, right: Counts) -> Counts:
    return Counts(
        ref=left.ref + right.ref,
        hyp=left.hyp + right.hyp,
        hits=left.hits + right.hits,
        sub=left.sub + right.sub,
        delete=left.delete + right.delete,
        insert=left.insert + right.insert,
    )


def main() -> int:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.tessdata_dir = args.tessdata_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args)
    csv_path = args.output_dir / f"{args.split_name}_per_file.csv"
    summary_path = args.output_dir / f"{args.split_name}_summary.txt"
    predictions_dir = args.output_dir / "predictions" / args.split_name
    predictions_dir.mkdir(parents=True, exist_ok=True)

    total_char = Counts()
    total_word = Counts()
    rows: list[dict[str, str]] = []

    for index, (image, truth_path) in enumerate(pairs, start=1):
        if index % 100 == 0 or index == 1:
            print(f"OCR {index}/{len(pairs)} {image.name}", flush=True)
        ref = normalize_text(truth_path.read_text(encoding="utf-8-sig"))
        hyp = run_ocr(args, image)
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
            }
        )

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    char_rates = rate_summary(total_char)
    word_rates = rate_summary(total_word)
    summary_lines = [
        f"dataset: {args.dataset}",
        f"tessdata_dir: {args.tessdata_dir}",
        f"language: {args.lang}",
        f"split: {args.split_name}",
        f"files: {len(pairs)}",
        "",
        f"CER_percent: {fmt(char_rates['error_rate'])}",
        f"char_recall_percent: {fmt(char_rates['recall'])}",
        f"char_precision_percent: {fmt(char_rates['precision'])}",
        f"char_f1_percent: {fmt(char_rates['f1'])}",
        f"chars_ref: {total_char.ref}",
        f"chars_hyp: {total_char.hyp}",
        f"chars_hits: {total_char.hits}",
        f"chars_sub: {total_char.sub}",
        f"chars_del: {total_char.delete}",
        f"chars_ins: {total_char.insert}",
        "",
        f"WER_percent: {fmt(word_rates['error_rate'])}",
        f"word_recall_percent: {fmt(word_rates['recall'])}",
        f"word_precision_percent: {fmt(word_rates['precision'])}",
        f"word_f1_percent: {fmt(word_rates['f1'])}",
        f"words_ref: {total_word.ref}",
        f"words_hyp: {total_word.hyp}",
        f"words_hits: {total_word.hits}",
        f"words_sub: {total_word.sub}",
        f"words_del: {total_word.delete}",
        f"words_ins: {total_word.insert}",
        "",
        f"per_file_csv: {csv_path}",
        f"predictions_dir: {predictions_dir}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
