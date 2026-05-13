#!/usr/bin/env python3
"""Fine-tune a PaddleOCR text-recognition model with PaddleX."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAIN_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
    "official_pretrained_model/arabic_PP-OCRv5_mobile_rec_pretrained.pdparams"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PaddleOCR text recognition.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("train") / "paddle_rec_dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("train") / "paddle_rec_arabic")
    parser.add_argument("--model", default="arabic_PP-OCRv5_mobile_rec")
    parser.add_argument("--config", type=Path, default=None, help="Optional PaddleX model config YAML.")
    parser.add_argument("--paddleocr-repo", type=Path, default=None, help="Path to a full PaddleOCR repo containing tools/train.py.")
    parser.add_argument("--pretrain-weight", default=DEFAULT_PRETRAIN_URL)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def configure_env(args: argparse.Namespace) -> None:
    cache_home = ROOT_DIR / ".cache" / "home"
    paddlex_cache = ROOT_DIR / ".cache" / "paddlex"
    cache_home.mkdir(parents=True, exist_ok=True)
    paddlex_cache.mkdir(parents=True, exist_ok=True)
    os.environ["USERPROFILE"] = str(cache_home)
    os.environ["HOME"] = str(cache_home)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(paddlex_cache)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if args.paddleocr_repo:
        os.environ["PADDLE_PDX_PADDLEOCR_PATH"] = str(args.paddleocr_repo.resolve())


def validate_args(args: argparse.Namespace) -> None:
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not (args.dataset_dir / "train.txt").exists():
        raise FileNotFoundError(f"Missing train.txt in {args.dataset_dir}")
    if not (args.dataset_dir / "val.txt").exists():
        raise FileNotFoundError(f"Missing val.txt in {args.dataset_dir}")
    if not (args.dataset_dir / "dict.txt").exists():
        raise FileNotFoundError(f"Missing dict.txt in {args.dataset_dir}")

    repo = Path(os.environ.get("PADDLE_PDX_PADDLEOCR_PATH", ""))
    if not repo or not (repo / "tools" / "train.py").exists():
        raise FileNotFoundError(
            "PaddleOCR training source is missing. Install or clone PaddleOCR, then pass "
            "--paddleocr-repo <path-to-PaddleOCR>. The installed paddleocr package only has inference APIs."
        )


def main() -> int:
    args = parse_args()
    configure_env(args)
    validate_args(args)

    import paddlex
    from paddlex.repo_apis.base import Config, PaddleModel

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.config
    if config_path is None:
        config_path = (
            Path(paddlex.__file__).resolve().parent
            / "configs"
            / "modules"
            / "text_recognition"
            / f"{args.model}.yaml"
        )
    if not config_path.exists():
        raise FileNotFoundError(f"Missing PaddleX config: {config_path}")

    config = Config(args.model, config_path=str(config_path))
    config.update_dataset(str(args.dataset_dir), "MSTextRecDataset")
    config.update_pretrained_weights(args.pretrain_weight)
    config.update_batch_size(args.batch_size)
    config.update_learning_rate(args.learning_rate)
    config._update_epochs(args.epochs)
    config.update_log_interval(args.log_interval)
    config.update_num_workers(args.num_workers, ["train", "eval"])
    config.update_device(args.device)
    config._update_output_dir(str(args.output_dir))
    config.dump(str(args.output_dir / "config.yaml"))

    print(f"dataset_dir: {args.dataset_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"model: {args.model}")
    print(f"device: {args.device}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"pretrain_weight: {args.pretrain_weight}")
    print(f"config: {config_path}")

    model = PaddleModel(config=config)
    result = model.train(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs_iters=args.epochs,
        device=args.device,
        num_workers=args.num_workers,
        use_vdl=False,
        save_dir=str(args.output_dir),
    )
    return int(result.returncode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
