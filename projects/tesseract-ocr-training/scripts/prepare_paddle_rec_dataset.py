#!/usr/bin/env python3
"""Prepare a PaddleOCR text-recognition dataset from image/text pairs."""

from __future__ import annotations

import argparse
import shutil
import unicodedata
from pathlib import Path


DEFAULT_DATASET = Path(r"E:\TRDG\new_ds_for_finetune\test_nrt_pdf_images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train.txt, val.txt, and dict.txt for PaddleOCR rec training.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("train") / "paddle_rec_dataset")
    parser.add_argument("--train-list", type=Path, default=Path("train") / "ara" / "lists" / "ara.training_files.txt")
    parser.add_argument("--val-list", type=Path, default=Path("train") / "ara" / "lists" / "ara.eval_files.txt")
    parser.add_argument("--image-ext", default=".png")
    parser.add_argument("--copy-images", action="store_true", help="Copy images into output-dir/images instead of using absolute paths.")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    return parser.parse_args()


def normalize_label(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    return " ".join(text.split())


def stems_from_list(path: Path) -> list[str]:
    stems: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            stems.append(Path(line).stem)
    return stems


def write_split(
    split_name: str,
    stems: list[str],
    dataset: Path,
    output_dir: Path,
    image_ext: str,
    copy_images: bool,
    limit: int,
) -> tuple[int, set[str]]:
    if limit > 0:
        stems = stems[:limit]
    chars: set[str] = set()
    lines: list[str] = []
    image_output_dir = output_dir / "images"
    if copy_images:
        image_output_dir.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        image = dataset / f"{stem}{image_ext}"
        truth = dataset / f"{stem}.txt"
        if not image.exists() or not truth.exists():
            continue
        label = normalize_label(truth.read_text(encoding="utf-8-sig"))
        if not label:
            continue
        for char in label:
            if char not in ("\t", "\n", "\r", " "):
                chars.add(char)
        if copy_images:
            rel_image = Path("images") / image.name
            shutil.copy2(image, output_dir / rel_image)
            image_ref = rel_image.as_posix()
        else:
            image_ref = str(image.resolve())
        lines.append(f"{image_ref}\t{label}")

    (output_dir / f"{split_name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines), chars


def main() -> int:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_stems = stems_from_list(args.train_list)
    val_stems = stems_from_list(args.val_list)
    train_count, train_chars = write_split(
        "train",
        train_stems,
        args.dataset,
        args.output_dir,
        args.image_ext,
        args.copy_images,
        args.limit_train,
    )
    val_count, val_chars = write_split(
        "val",
        val_stems,
        args.dataset,
        args.output_dir,
        args.image_ext,
        args.copy_images,
        args.limit_val,
    )
    chars = sorted(train_chars | val_chars)
    (args.output_dir / "dict.txt").write_text("\n".join(chars) + "\n", encoding="utf-8")
    summary = [
        f"dataset: {args.dataset}",
        f"output_dir: {args.output_dir}",
        f"train_samples: {train_count}",
        f"val_samples: {val_count}",
        f"dict_chars: {len(chars)}",
        f"copy_images: {args.copy_images}",
    ]
    (args.output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
