"""Train a local Surya-style text/layout region detector from JSON annotations."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.ops import box_iou

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback before requirements install.
    def tqdm(iterable: Any, **_: Any) -> Any:
        """Return the iterable unchanged when tqdm is unavailable."""
        return iterable

try:
    from .utils.region_model import (
        CATEGORY_ID_TO_NAME,
        NUM_REGION_CLASSES,
        PerImageJsonRegionDataset,
        build_region_detector,
        collate_detection_batch,
        move_targets_to_device,
        save_checkpoint,
    )
except ImportError:  # pragma: no cover - used when run as python train_region_detector.py.
    from utils.region_model import (  # type: ignore
        CATEGORY_ID_TO_NAME,
        NUM_REGION_CLASSES,
        PerImageJsonRegionDataset,
        build_region_detector,
        collate_detection_batch,
        move_targets_to_device,
        save_checkpoint,
    )

logger = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT = Path(r"E:\Antigravity_Code\get_pdfs_KCAC\kcac_client\dataset\409")


def parse_args() -> argparse.Namespace:
    """Parse command-line training options."""
    parser = argparse.ArgumentParser(description="Train a local Surya-style region detector")
    parser.add_argument("--train-image-dir", type=Path, default=DEFAULT_DATASET_ROOT / "pages")
    parser.add_argument("--train-anno-dir", type=Path, default=DEFAULT_DATASET_ROOT / "json_of_pages" / "annotations" / "per_image")
    parser.add_argument("--val-image-dir", type=Path, default=DEFAULT_DATASET_ROOT / "test" / "img")
    parser.add_argument("--val-anno-dir", type=Path, default=DEFAULT_DATASET_ROOT / "test" / "anno")
    parser.add_argument("--out-dir", type=Path, default=Path("results") / "training" / "surya_like_region_detector")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--score-thresh", type=float, default=0.35)
    parser.add_argument("--iou-thresh", type=float, default=0.50)
    parser.add_argument("--min-conf", type=float, default=0.40)
    parser.add_argument("--min-layout-score", type=float, default=0.30)
    parser.add_argument("--target", choices=["text_lines", "layout", "both"], default="both")
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not initialize from COCO Faster R-CNN weights")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true", help="Load --resume and evaluate without training")
    return parser.parse_args()


def setup_logging(out_dir: Path) -> Path:
    """Configure stream and file logging for training."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    return log_path


def set_seed(seed: int) -> None:
    """Set Python and Torch seeds for repeatable train/validation splits."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(image_dir: Path, anno_dir: Path, args: argparse.Namespace, max_images: int | None) -> PerImageJsonRegionDataset:
    """Build a per-image JSON dataset according to the selected target type."""
    return PerImageJsonRegionDataset(
        image_dir=image_dir,
        anno_dir=anno_dir,
        include_text_lines=args.target in {"text_lines", "both"},
        include_layout=args.target in {"layout", "both"},
        min_conf=args.min_conf,
        min_score=args.min_layout_score,
        max_images=max_images,
    )


def split_dataset(dataset: Dataset[Any], val_ratio: float = 0.15, seed: int = 42) -> tuple[Dataset[Any], Dataset[Any]]:
    """Split one dataset into train and validation subsets if no explicit validation set exists."""
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(len(indices) * val_ratio)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def make_loader(dataset: Dataset[Any], batch_size: int, workers: int, shuffle: bool) -> DataLoader[Any]:
    """Create a detection DataLoader with a torchvision-compatible collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection_batch,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    epoch: int,
    use_amp: bool,
) -> dict[str, float]:
    """Run one training epoch and return averaged loss metrics."""
    model.train()
    metric_sums: dict[str, float] = {}
    steps = 0
    start = time.time()
    progress = tqdm(loader, desc=f"Epoch {epoch} train", unit="batch")
    for images, targets in progress:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        steps += 1
        for key, value in loss_dict.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value.detach().cpu())
        metric_sums["loss_total"] = metric_sums.get("loss_total", 0.0) + float(losses.detach().cpu())
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{float(losses.detach().cpu()):.4f}")

    elapsed = time.time() - start
    metrics = {key: value / max(steps, 1) for key, value in metric_sums.items()}
    metrics["seconds"] = elapsed
    return metrics


def greedy_match_class(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresh: float,
) -> tuple[int, int, int]:
    """Greedily match predictions to ground truth for one class."""
    if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
        return 0, 0, 0
    if pred_boxes.numel() == 0:
        return 0, 0, int(gt_boxes.shape[0])
    if gt_boxes.numel() == 0:
        return 0, int(pred_boxes.shape[0]), 0

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]
    ious = box_iou(pred_boxes, gt_boxes)
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    for pred_idx in range(pred_boxes.shape[0]):
        best_iou = 0.0
        best_gt = -1
        for gt_idx in range(gt_boxes.shape[0]):
            if gt_idx in matched_gt:
                continue
            iou = float(ious[pred_idx, gt_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_iou >= iou_thresh and best_gt >= 0:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1
    fn = int(gt_boxes.shape[0]) - len(matched_gt)
    return tp, fp, fn


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    score_thresh: float,
    iou_thresh: float,
) -> dict[str, Any]:
    """Evaluate detection quality with class-aware IoU matching."""
    model.eval()
    totals = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in range(1, NUM_REGION_CLASSES)}
    progress = tqdm(loader, desc="Validation", unit="batch")
    for images, targets in progress:
        images = [image.to(device, non_blocking=True) for image in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            boxes = output["boxes"].detach().cpu()
            labels = output["labels"].detach().cpu()
            scores = output["scores"].detach().cpu()
            keep = scores >= score_thresh
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            gt_boxes_all = target["boxes"].detach().cpu()
            gt_labels_all = target["labels"].detach().cpu()
            for class_id in range(1, NUM_REGION_CLASSES):
                pred_mask = labels == class_id
                gt_mask = gt_labels_all == class_id
                tp, fp, fn = greedy_match_class(boxes[pred_mask], scores[pred_mask], gt_boxes_all[gt_mask], iou_thresh)
                totals[class_id]["tp"] += tp
                totals[class_id]["fp"] += fp
                totals[class_id]["fn"] += fn

    tp_total = sum(item["tp"] for item in totals.values())
    fp_total = sum(item["fp"] for item in totals.values())
    fn_total = sum(item["fn"] for item in totals.values())
    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    per_class: dict[str, dict[str, float | int]] = {}
    for class_id, counts in totals.items():
        c_tp = counts["tp"]
        c_fp = counts["fp"]
        c_fn = counts["fn"]
        c_precision = c_tp / max(c_tp + c_fp, 1)
        c_recall = c_tp / max(c_tp + c_fn, 1)
        c_f1 = 0.0 if c_precision + c_recall == 0.0 else 2.0 * c_precision * c_recall / (c_precision + c_recall)
        per_class[CATEGORY_ID_TO_NAME[class_id]] = {
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
            "precision": round(c_precision, 4),
            "recall": round(c_recall, 4),
            "f1": round(c_f1, 4),
        }
    return {
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_class": per_class,
    }


def load_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path | None,
    device: torch.device,
) -> tuple[int, float]:
    """Load model and optimizer state from a resume checkpoint if provided."""
    if checkpoint_path is None:
        return 1, 0.0
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_f1 = float(checkpoint.get("best_f1", 0.0))
    logger.info("Resumed %s at epoch %d with best_f1 %.4f", checkpoint_path, start_epoch, best_f1)
    return start_epoch, best_f1


def save_history(history: list[dict[str, Any]], out_dir: Path) -> Path:
    """Save epoch-by-epoch training history as JSON."""
    out_path = Path(out_dir) / "history.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
    return out_path


def save_eval_report(metrics: dict[str, Any], out_dir: Path) -> Path:
    """Save an evaluation-only report as JSON."""
    out_path = Path(out_dir) / "eval_report.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return out_path


def main() -> None:
    """Train, validate, and save last/best region detector checkpoints."""
    args = parse_args()
    log_path = setup_logging(args.out_dir)
    set_seed(args.seed)
    logger.info("Logging to %s", log_path)
    logger.info("Training target: %s", args.target)

    train_dataset = build_dataset(args.train_image_dir, args.train_anno_dir, args, args.max_train_images)
    if args.val_image_dir.exists() and args.val_anno_dir.exists():
        val_dataset: Dataset[Any] = build_dataset(args.val_image_dir, args.val_anno_dir, args, args.max_val_images)
    else:
        logger.info("Validation dirs not found; splitting train data")
        train_dataset, val_dataset = split_dataset(train_dataset, seed=args.seed)

    train_loader = make_loader(train_dataset, args.batch_size, args.workers, shuffle=True)
    val_loader = make_loader(val_dataset, max(1, min(args.batch_size, 2)), args.workers, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    model = build_region_detector(
        num_classes=NUM_REGION_CLASSES,
        pretrained=not args.no_pretrained,
        min_size=args.min_size,
        max_size=args.max_size,
    ).to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    start_epoch, best_f1 = load_resume(model, optimizer, args.resume, device)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    if args.eval_only:
        if args.resume is None:
            raise SystemExit("--eval-only requires --resume CHECKPOINT")
        val_metrics = evaluate_model(model, val_loader, device, args.score_thresh, args.iou_thresh)
        report_path = save_eval_report(val_metrics, args.out_dir)
        logger.info(
            "Eval only | F1 %.4f | P %.4f | R %.4f | report %s",
            val_metrics["f1"],
            val_metrics["precision"],
            val_metrics["recall"],
            report_path,
        )
        return

    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, scaler, epoch, use_amp)
        val_metrics = evaluate_model(model, val_loader, device, args.score_thresh, args.iou_thresh)
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        logger.info(
            "Epoch %d/%d | loss %.4f | val F1 %.4f | P %.4f | R %.4f",
            epoch,
            args.epochs,
            train_metrics.get("loss_total", 0.0),
            val_metrics["f1"],
            val_metrics["precision"],
            val_metrics["recall"],
        )

        checkpoint_args = vars(args).copy()
        checkpoint_args["category_id_to_name"] = CATEGORY_ID_TO_NAME
        last_path = save_checkpoint(Path(args.out_dir) / "last_model.pth", model, optimizer, epoch, best_f1, checkpoint_args)
        if float(val_metrics["f1"]) >= best_f1:
            best_f1 = float(val_metrics["f1"])
            best_path = save_checkpoint(Path(args.out_dir) / "best_model.pth", model, optimizer, epoch, best_f1, checkpoint_args)
            logger.info("Saved new best model to %s", best_path)
        logger.info("Saved last model to %s", last_path)
        save_history(history, args.out_dir)

    logger.info("Training complete. Best validation F1: %.4f", best_f1)
    logger.info("Load for prediction with: python predict_region_detector.py --checkpoint %s --image-dir PATH", Path(args.out_dir) / "best_model.pth")


if __name__ == "__main__":
    main()
