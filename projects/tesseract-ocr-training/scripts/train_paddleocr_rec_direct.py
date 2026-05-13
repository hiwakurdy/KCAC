#!/usr/bin/env python3
'''
    & "C:\Users\Hiwa\.conda\envs\paddleocr_eval\python.exe" `
        scripts\train_paddleocr_rec_direct.py --dataset-dir train\paddle_rec_dataset `
        --output-dir train\paddleocr_rec_arabic_v3_line_w1024_e1 --epochs 1 `
        --batch-size 1 --learning-rate 0.0001 --device gpu:0 --num-workers 0 `
        --max-text-length 220 --image-width 1024 --eval-interval 500 --print-interval 100
'''
"""Train PaddleOCR text recognition directly with tools/train.py."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPO = Path(
    r"C:\Users\Hiwa\.conda\envs\paddleocr_eval\Lib\site-packages\paddlex\repo_manager\repos\PaddleOCR"
)
DEFAULT_BASE_CONFIG = DEFAULT_REPO / "configs" / "rec" / "PP-OCRv3" / "multi_language" / "arabic_PP-OCRv3_mobile_rec.yml"
DEFAULT_PRETRAIN = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
    "official_pretrained_model/arabic_PP-OCRv3_mobile_rec_pretrained.pdparams"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PaddleOCR Arabic text recognition.")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, default=Path("train") / "paddle_rec_dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("train") / "paddleocr_rec_arabic_v3")
    parser.add_argument("--pretrain", default=DEFAULT_PRETRAIN)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-text-length", type=int, default=220)
    parser.add_argument("--image-width", type=int, default=3200)
    parser.add_argument("--image-height", type=int, default=48)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--print-interval", type=int, default=5)
    return parser.parse_args()


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target: Any = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def patch_transforms(transforms: list[dict[str, Any]], image_shape_chw: list[int]) -> None:
    image_shape_hwc = [image_shape_chw[1], image_shape_chw[2], image_shape_chw[0]]
    for transform in transforms:
        if "RecResizeImg" in transform:
            transform["RecResizeImg"]["image_shape"] = image_shape_chw
        if "RecConAug" in transform:
            transform["RecConAug"]["image_shape"] = image_shape_hwc


def patch_sar_length(config: dict[str, Any], max_text_length: int) -> None:
    head = config.get("Architecture", {}).get("Head", {})
    for item in head.get("head_list", []):
        sar = item.get("SARHead")
        if isinstance(sar, dict) and "max_text_length" in sar:
            sar["max_text_length"] = max_text_length


def build_config(args: argparse.Namespace) -> Path:
    args.repo = args.repo.resolve()
    args.base_config = args.base_config.resolve()
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not (args.repo / "tools" / "train.py").exists():
        raise FileNotFoundError(f"PaddleOCR tools/train.py not found in {args.repo}")
    for name in ("train.txt", "val.txt", "dict.txt"):
        if not (args.dataset_dir / name).exists():
            raise FileNotFoundError(f"Missing {name} in {args.dataset_dir}")

    config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    image_shape = [3, args.image_height, args.image_width]
    updates = {
        "Global.epoch_num": args.epochs,
        "Global.print_batch_step": args.print_interval,
        "Global.save_model_dir": str(args.output_dir),
        "Global.save_epoch_step": 1,
        "Global.eval_batch_step": [0, args.eval_interval],
        "Global.pretrained_model": args.pretrain,
        "Global.checkpoints": None,
        "Global.character_dict_path": str(args.dataset_dir / "dict.txt"),
        "Global.max_text_length": args.max_text_length,
        "Global.use_space_char": True,
        "Global.use_gpu": args.device.lower().startswith("gpu"),
        "Optimizer.lr.learning_rate": args.learning_rate,
        "Train.dataset.data_dir": str(args.dataset_dir),
        "Train.dataset.label_file_list": [str(args.dataset_dir / "train.txt")],
        "Train.loader.batch_size_per_card": args.batch_size,
        "Train.loader.drop_last": False,
        "Train.loader.num_workers": args.num_workers,
        "Eval.dataset.data_dir": str(args.dataset_dir),
        "Eval.dataset.label_file_list": [str(args.dataset_dir / "val.txt")],
        "Eval.loader.batch_size_per_card": args.batch_size,
        "Eval.loader.num_workers": args.num_workers,
    }
    for key, value in updates.items():
        set_nested(config, key, value)
    patch_sar_length(config, args.max_text_length)
    patch_transforms(config["Train"]["dataset"]["transforms"], image_shape)
    patch_transforms(config["Eval"]["dataset"]["transforms"], image_shape)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "train_config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config_path


def main() -> int:
    args = parse_args()
    config_path = build_config(args)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PADDLE_OCR_BASE_DIR"] = str((ROOT_DIR / "train" / "paddleocr_cache").resolve())
    if args.device.lower().startswith("gpu") and ":" in args.device:
        env["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]

    print(f"repo: {args.repo.resolve()}")
    print(f"config: {config_path}")
    print(f"dataset_dir: {args.dataset_dir.resolve()}")
    print(f"output_dir: {args.output_dir.resolve()}")
    print(f"pretrain: {args.pretrain}")
    print(f"device: {args.device}")

    cmd = [sys.executable, "tools/train.py", "-c", str(config_path)]
    proc = subprocess.run(cmd, cwd=args.repo, env=env)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
