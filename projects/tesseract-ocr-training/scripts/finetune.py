#!/usr/bin/env python3
"""Build LSTM training files and fine-tune with Tesseract."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re


DEFAULT_DATASET = Path(r"E:\TRDG\new_ds_for_finetune\test_nrt_pdf_images")
DEFAULT_WORKDIR = Path("train") / "urd"


@dataclass(frozen=True)
class Pair:
    image: Path
    text: Path
    stem: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and fine-tune Tesseract Urdu LSTM training data."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--base-lang", default="urd")
    parser.add_argument("--output-lang", default="urd")
    parser.add_argument("--tesseract", default=shutil.which("tesseract") or "tesseract")
    parser.add_argument(
        "--lstmtraining", default=shutil.which("lstmtraining") or "lstmtraining"
    )
    parser.add_argument(
        "--combine-tessdata",
        default=shutil.which("combine_tessdata") or "combine_tessdata",
    )
    parser.add_argument("--base-traineddata", type=Path, default=None)
    parser.add_argument("--image-ext", default=".png")
    parser.add_argument("--psm", type=int, default=7)
    parser.add_argument(
        "--fallback-psm",
        default="13",
        help="Comma-separated PSM values to try when the primary PSM fails.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0, help="Use N pairs; 0 means all.")
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--force-lstmf", action="store_true")
    parser.add_argument(
        "--keep-unsupported",
        action="store_true",
        help="Do not map/drop characters missing from the base model unicharset.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--make-lstmf-only", action="store_true")
    parser.add_argument("--skip-lstmf", action="store_true")
    parser.add_argument(
        "--existing-lstmf-only",
        action="store_true",
        help="Train only from .lstmf files that already exist in the workdir.",
    )
    return parser.parse_args()


def quote_for_log(value: str) -> str:
    if any(ch.isspace() for ch in value):
        return f'"{value}"'
    return value


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def run(args: list[str], log_path: Path, check: bool = True) -> int:
    command = " ".join(quote_for_log(arg) for arg in args)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {command}\n")
        log.flush()
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        log.write(proc.stdout)
        log.flush()
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {command}")
    return proc.returncode


def psm_values(args: argparse.Namespace) -> list[int]:
    values = [args.psm]
    for raw in args.fallback_psm.split(","):
        raw = raw.strip()
        if raw:
            value = int(raw)
            if value not in values:
                values.append(value)
    return values


def resolve_base_traineddata(args: argparse.Namespace) -> Path:
    if args.base_traineddata:
        return args.base_traineddata.resolve()

    local_best = Path("train") / "base_models" / f"{args.base_lang}.traineddata"
    if local_best.exists():
        return local_best.resolve()

    tesseract_path = Path(args.tesseract)
    if tesseract_path.exists():
        candidate = tesseract_path.parent / "tessdata" / f"{args.base_lang}.traineddata"
        if candidate.exists():
            return candidate.resolve()

    candidate = Path(r"C:\Program Files\Tesseract-OCR\tessdata") / (
        f"{args.base_lang}.traineddata"
    )
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find {args.base_lang}.traineddata. Pass --base-traineddata."
    )


def resolve_installed_tessdata(args: argparse.Namespace) -> Path:
    tesseract_path = Path(args.tesseract)
    if tesseract_path.exists():
        candidate = tesseract_path.parent / "tessdata"
        if (candidate / "configs" / "lstm.train").exists():
            return candidate.resolve()

    candidate = Path(r"C:\Program Files\Tesseract-OCR\tessdata")
    if (candidate / "configs" / "lstm.train").exists():
        return candidate.resolve()

    raise FileNotFoundError("Could not find installed tessdata/configs/lstm.train.")


def link_or_copy(source: Path, dest: Path, copy_only: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    if copy_only:
        shutil.copy2(source, dest)
        return
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def prepare_lstmf_tessdata(
    args: argparse.Namespace, base_traineddata: Path, workdir: Path
) -> Path:
    installed_tessdata = resolve_installed_tessdata(args)
    run_tessdata = workdir / "tessdata"
    run_tessdata.mkdir(parents=True, exist_ok=True)
    (run_tessdata / "configs").mkdir(parents=True, exist_ok=True)

    staged_model = run_tessdata / f"{args.base_lang}.traineddata"
    if staged_model.exists() and staged_model.stat().st_size != base_traineddata.stat().st_size:
        staged_model.unlink()
    link_or_copy(base_traineddata, staged_model)
    shutil.copy2(installed_tessdata / "configs" / "lstm.train", run_tessdata / "configs")
    return run_tessdata.resolve()


def extract_base_components(
    args: argparse.Namespace, base_traineddata: Path, base_dir: Path, log_path: Path
) -> tuple[Path, Path]:
    prefix = base_dir / f"{safe_name(args.base_lang)}."
    run([args.combine_tessdata, "-u", str(base_traineddata), str(prefix)], log_path)
    base_lstm = base_dir / f"{safe_name(args.base_lang)}.lstm"
    unicharset = base_dir / f"{safe_name(args.base_lang)}.lstm-unicharset"
    if not base_lstm.exists() or not unicharset.exists():
        raise FileNotFoundError("Could not extract base LSTM/unicharset components.")
    return base_lstm.resolve(), unicharset.resolve()


def load_unicharset(path: Path) -> set[str]:
    chars: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if line:
            chars.add(line.split(" ")[0])
    return chars


def find_pairs(dataset: Path, image_ext: str) -> list[Pair]:
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset}")
    pairs: list[Pair] = []
    for image in sorted(dataset.glob(f"*{image_ext}")):
        text = image.with_suffix(".txt")
        if text.exists():
            pairs.append(Pair(image=image.resolve(), text=text.resolve(), stem=image.stem))
    if not pairs:
        raise ValueError(f"No {image_ext}/.txt pairs found in {dataset}")
    return pairs


def split_pairs(
    pairs: list[Pair], eval_ratio: float, limit: int, seed: int
) -> tuple[list[Pair], list[Pair]]:
    selected = list(pairs)
    if limit > 0:
        selected = selected[:limit]
    rng = random.Random(seed)
    rng.shuffle(selected)
    if len(selected) < 2:
        return selected, []
    eval_count = max(1, int(round(len(selected) * eval_ratio)))
    eval_count = min(eval_count, len(selected) - 1)
    return selected[eval_count:], selected[:eval_count]


def describe_char(char: str) -> str:
    name = unicodedata.name(char, "UNKNOWN")
    return f"U+{ord(char):04X} {char} {name}"


def read_line_text(
    path: Path, allowed_chars: set[str] | None, stats: Counter[str]
) -> str:
    text = path.read_text(encoding="utf-8-sig").strip()
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u060c": ",",
        "\u0640": "",
        "\u06a4": "\u0641",
        "\u2044": "/",
        "\u2153": "1/3",
    }

    output: list[str] = []
    for char in text:
        replacement = replacements.get(char, char)
        if replacement != char:
            stats[f"map {describe_char(char)} -> {replacement or '<drop>'}"] += 1
        for item in replacement:
            if item.isspace():
                output.append(" ")
            elif allowed_chars is None or item in allowed_chars:
                output.append(item)
            else:
                stats[f"drop {describe_char(item)}"] += 1

    return " ".join("".join(output).split())


def read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as image_file:
        header = image_file.read(24)
    png_signature = b"\x89PNG\r\n\x1a\n"
    if len(header) >= 24 and header[:8] == png_signature and header[12:16] == b"IHDR":
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        return width, height
    raise ValueError(f"Can only read PNG dimensions directly: {path}")


def make_line_box_text(text: str, width: int, height: int) -> str:
    lines: list[str] = []
    if text:
        normalized = unicodedata.normalize("NFC", text)
        for index in range(1, len(normalized)):
            char = normalized[index]
            prev_char = normalized[index - 1]
            if unicodedata.combining(char):
                lines.append(f"{prev_char + char} 0 0 {width} {height} 0")
            elif not unicodedata.combining(prev_char):
                lines.append(f"{prev_char} 0 0 {width} {height} 0")
        if not unicodedata.combining(normalized[-1]):
            lines.append(f"{normalized[-1]} 0 0 {width} {height} 0")
    lines.append(f"\t 0 0 {width} {height} 0")
    return "\n".join(lines) + "\n"


def stage_image(pair: Pair, gt_base: Path, copy_images: bool) -> Path:
    staged_image = gt_base.with_suffix(pair.image.suffix.lower())
    link_or_copy(pair.image, staged_image, copy_only=copy_images)
    return staged_image


def write_ground_truth(
    pair: Pair,
    gt_dir: Path,
    copy_images: bool,
    allowed_chars: set[str] | None,
    stats: Counter[str],
) -> tuple[Path, Path]:
    gt_base = gt_dir / pair.stem
    staged_image = stage_image(pair, gt_base, copy_images)
    text = read_line_text(pair.text, allowed_chars, stats)
    width, height = read_png_size(staged_image)
    box_text = make_line_box_text(text, width, height)
    (gt_base.with_suffix(".gt.txt")).write_text(text + "\n", encoding="utf-8")
    (gt_base.with_suffix(".box")).write_text(box_text, encoding="utf-8")
    return gt_base, staged_image


def make_lstmf(
    args: argparse.Namespace,
    pairs: list[Pair],
    gt_dir: Path,
    tessdata_dir: Path,
    log_path: Path,
    allowed_chars: set[str] | None,
    stats: Counter[str],
) -> list[Path]:
    lstmf_paths: list[Path] = []
    for index, pair in enumerate(pairs, start=1):
        gt_base, staged_image = write_ground_truth(
            pair, gt_dir, args.copy_images, allowed_chars, stats
        )
        lstmf = gt_base.with_suffix(".lstmf")
        if args.force_lstmf and lstmf.exists():
            lstmf.unlink()
        if not lstmf.exists():
            last_code = 0
            for psm in psm_values(args):
                last_code = run(
                    [
                        args.tesseract,
                        str(staged_image),
                        str(gt_base),
                        "--psm",
                        str(psm),
                        "--dpi",
                        str(args.dpi),
                        "-l",
                        args.base_lang,
                        "--tessdata-dir",
                        str(tessdata_dir),
                        "lstm.train",
                    ],
                    log_path,
                    check=False,
                )
                if lstmf.exists():
                    break
            if not lstmf.exists() and last_code != 0:
                raise RuntimeError(f"Could not create lstmf for {pair.stem}")
        if not lstmf.exists():
            raise FileNotFoundError(f"Expected lstmf was not created: {lstmf}")
        lstmf_paths.append(lstmf.resolve())
        if index % 250 == 0:
            print(f"created/checked {index} lstmf files", flush=True)
    return lstmf_paths


def write_list(path: Path, items: list[Path]) -> None:
    rows = [str(item.resolve()).replace("\\", "/") for item in items]
    with path.open("w", encoding="utf-8", newline="\n") as list_file:
        list_file.write("\n".join(rows) + "\n")


def find_checkpoint(prefix: Path) -> Path:
    best_pattern = re.compile(rf"^{re.escape(prefix.name)}_([0-9.]+)_.*\.checkpoint$")
    best: list[tuple[float, Path]] = []
    for candidate in prefix.parent.glob(f"{prefix.name}_*.checkpoint"):
        match = best_pattern.match(candidate.name)
        if match:
            best.append((float(match.group(1)), candidate))
    if best:
        return min(best, key=lambda item: item[0])[1]

    direct = Path(str(prefix) + "_checkpoint")
    if direct.exists():
        return direct
    candidates = sorted(
        prefix.parent.glob("*.checkpoint"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No checkpoint found for model output prefix {prefix}")


def write_normalization_report(path: Path, stats: Counter[str]) -> None:
    if not stats:
        path.write_text("No text normalization changes were needed.\n", encoding="utf-8")
        return
    lines = ["Text normalization changes:", ""]
    for item, count in stats.most_common():
        lines.append(f"{count}\t{item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    workdir = args.workdir.resolve()
    gt_dir = workdir / "ground-truth"
    list_dir = workdir / "lists"
    base_dir = workdir / "base"
    checkpoint_dir = workdir / "checkpoints"
    output_dir = workdir / "output"
    log_path = workdir / "train.log"

    for path in (gt_dir, list_dir, base_dir, checkpoint_dir, output_dir):
        path.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    base_traineddata = resolve_base_traineddata(args)
    tessdata_dir = prepare_lstmf_tessdata(args, base_traineddata, workdir)
    base_lstm, unicharset_path = extract_base_components(
        args, base_traineddata, base_dir, log_path
    )
    allowed_chars = None if args.keep_unsupported else load_unicharset(unicharset_path)
    normalization_stats: Counter[str] = Counter()
    pairs = find_pairs(dataset, args.image_ext)
    train_pairs, eval_pairs = split_pairs(pairs, args.eval_ratio, args.limit, args.seed)
    if args.existing_lstmf_only:
        train_pairs = [
            pair for pair in train_pairs if (gt_dir / f"{pair.stem}.lstmf").exists()
        ]
        eval_pairs = [
            pair for pair in eval_pairs if (gt_dir / f"{pair.stem}.lstmf").exists()
        ]
        if not eval_pairs and len(train_pairs) > 1:
            eval_pairs = train_pairs[-max(1, len(train_pairs) // 20) :]
            train_pairs = train_pairs[: -len(eval_pairs)]
    selected_pairs = train_pairs + eval_pairs

    print(f"dataset pairs: {len(pairs)}")
    print(f"selected pairs: {len(selected_pairs)}")
    print(f"train pairs: {len(train_pairs)}")
    print(f"eval pairs: {len(eval_pairs)}")
    print(f"workdir: {workdir}")
    print(f"base traineddata: {base_traineddata}")
    print(f"lstmf tessdata: {tessdata_dir}")
    print(f"base lstm: {base_lstm}")
    print(f"base unicharset: {unicharset_path}")

    if args.prepare_only:
        for pair in selected_pairs:
            write_ground_truth(
                pair, gt_dir, args.copy_images, allowed_chars, normalization_stats
            )
        write_normalization_report(workdir / "normalization_report.txt", normalization_stats)
        print("prepared ground-truth files only")
        return 0

    if args.skip_lstmf or args.existing_lstmf_only:
        train_lstmf = [gt_dir / f"{pair.stem}.lstmf" for pair in train_pairs]
        eval_lstmf = [gt_dir / f"{pair.stem}.lstmf" for pair in eval_pairs]
    else:
        train_lstmf = make_lstmf(
            args,
            train_pairs,
            gt_dir,
            tessdata_dir,
            log_path,
            allowed_chars,
            normalization_stats,
        )
        eval_lstmf = make_lstmf(
            args,
            eval_pairs,
            gt_dir,
            tessdata_dir,
            log_path,
            allowed_chars,
            normalization_stats,
        )
    write_normalization_report(workdir / "normalization_report.txt", normalization_stats)

    train_list = list_dir / f"{args.output_lang}.training_files.txt"
    eval_list = list_dir / f"{args.output_lang}.eval_files.txt"
    write_list(train_list, train_lstmf)
    write_list(eval_list, eval_lstmf)
    print(f"train list: {train_list}")
    print(f"eval list: {eval_list}")

    if args.make_lstmf_only:
        print("created lstmf/list files only")
        return 0

    model_prefix = checkpoint_dir / args.output_lang
    run(
        [
            args.lstmtraining,
            "--model_output",
            str(model_prefix),
            "--continue_from",
            str(base_lstm),
            "--traineddata",
            str(base_traineddata),
            "--train_listfile",
            str(train_list),
            "--eval_listfile",
            str(eval_list),
            "--max_iterations",
            str(args.max_iterations),
        ],
        log_path,
    )

    checkpoint = find_checkpoint(model_prefix)
    output_traineddata = output_dir / f"{args.output_lang}.traineddata"
    run(
        [
            args.lstmtraining,
            "--stop_training",
            "--continue_from",
            str(checkpoint),
            "--traineddata",
            str(base_traineddata),
            "--model_output",
            str(output_traineddata),
        ],
        log_path,
    )

    print(f"final model: {output_traineddata}")
    print(f"log: {log_path}")
    print(
        "test command: "
        f'tesseract IMAGE stdout -l {args.output_lang} --tessdata-dir "{output_dir}"'
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
