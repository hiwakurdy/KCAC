"""Evaluation and decision-gate helpers for Surya text-region predictions."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .. import config
except ImportError:  # pragma: no cover - used when scripts run from this directory.
    import config  # type: ignore

logger = logging.getLogger(__name__)

GOOD_FLAG = "\u2705 GOOD"
MEDIUM_FLAG = "\u26a0\ufe0f  MEDIUM"
POOR_FLAG = "\u274c POOR"


def _prediction_items(prediction: Any, key: str = "bboxes") -> list[Any]:
    """Return a list of bbox-like items from object-style or dict-style Surya output."""
    if prediction is None:
        return []
    if isinstance(prediction, dict):
        value = prediction.get(key)
        if value is None and key == "bboxes":
            value = prediction.get("text_lines") or prediction.get("layout") or []
        return list(value or [])
    return list(getattr(prediction, key, []) or [])


def _item_bbox(item: Any) -> list[float]:
    """Extract a [x1, y1, x2, y2] bbox from a prediction item."""
    if isinstance(item, dict):
        bbox = item.get("bbox") or item.get("box") or item.get("rectangle") or [0, 0, 0, 0]
    else:
        bbox = getattr(item, "bbox", [0, 0, 0, 0])
    return [float(v) for v in list(bbox)[:4]]


def _item_confidence(item: Any) -> float:
    """Extract a confidence-like score from a prediction item."""
    if isinstance(item, dict):
        value = item.get("confidence", item.get("score", 0.0))
    else:
        value = getattr(item, "confidence", getattr(item, "score", 0.0))
    return float(value or 0.0)


def _item_label(item: Any) -> str:
    """Extract a layout label from a prediction item."""
    if isinstance(item, dict):
        return str(item.get("label", item.get("name", "unknown")) or "unknown")
    return str(getattr(item, "label", "unknown") or "unknown")


def compute_text_coverage(
    bboxes: Sequence[Any],
    img_w: int,
    img_h: int,
    conf_thresh: float = config.CONF_THRESH,
) -> float:
    """Compute percent image area covered by accepted text boxes without double-counting overlaps."""
    if img_w <= 0 or img_h <= 0:
        return 0.0

    mask = np.zeros((int(img_h), int(img_w)), dtype=np.uint8)
    for bbox_item in bboxes:
        if _item_confidence(bbox_item) < conf_thresh:
            continue
        x1, y1, x2, y2 = _item_bbox(bbox_item)
        ix1 = max(0, min(int(round(x1)), img_w))
        iy1 = max(0, min(int(round(y1)), img_h))
        ix2 = max(0, min(int(round(x2)), img_w))
        iy2 = max(0, min(int(round(y2)), img_h))
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = 1

    return float(mask.sum()) / float(img_w * img_h) * 100.0


def quality_flag(n_lines: int, avg_conf: float, coverage_pct: float) -> str:
    """Assign GOOD, MEDIUM, or POOR based on configured confidence, count, and coverage gates."""
    if avg_conf >= config.GATE_GOOD and n_lines >= config.MIN_LINES and coverage_pct >= config.MIN_COV_PCT:
        return GOOD_FLAG
    if avg_conf >= config.GATE_MEDIUM and n_lines >= 1:
        return MEDIUM_FLAG
    return POOR_FLAG


def evaluate_single(
    name: str,
    img_w: int,
    img_h: int,
    det_pred: Any,
    lay_pred: Any,
    conf_thresh: float = config.CONF_THRESH,
) -> dict[str, Any]:
    """Evaluate one image prediction and return metrics used by the decision gate."""
    det_items = _prediction_items(det_pred)
    lay_items = _prediction_items(lay_pred)
    filtered_lines = [item for item in det_items if _item_confidence(item) >= conf_thresh]
    confidences = [_item_confidence(item) for item in filtered_lines]
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0
    max_confidence = float(np.max(confidences)) if confidences else 0.0
    coverage_pct = compute_text_coverage(det_items, img_w, img_h, conf_thresh)

    layout_types: dict[str, int] = {}
    for layout_item in lay_items:
        label = _item_label(layout_item)
        layout_types[label] = layout_types.get(label, 0) + 1

    flag = quality_flag(len(filtered_lines), avg_confidence, coverage_pct)
    return {
        "image": name,
        "width": int(img_w),
        "height": int(img_h),
        "total_det_bboxes": len(det_items),
        "filtered_lines": len(filtered_lines),
        "avg_confidence": round(avg_confidence, 4),
        "max_confidence": round(max_confidence, 4),
        "text_coverage_pct": round(coverage_pct, 2),
        "layout_types": layout_types,
        "layout_region_count": len(lay_items),
        "quality_flag": flag,
    }


def evaluate_batch(
    names: list[str],
    sizes: list[tuple[int, int]],
    det_preds: list[Any],
    lay_preds: list[Any],
    conf_thresh: float = config.CONF_THRESH,
) -> list[dict[str, Any]]:
    """Evaluate all prediction results in image order."""
    return [
        evaluate_single(name, width, height, det_pred, lay_pred, conf_thresh)
        for name, (width, height), det_pred, lay_pred in zip(names, sizes, det_preds, lay_preds)
    ]


def print_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Print a per-image table, summary counts, and a dataset-level decision verdict."""
    total = len(results)
    denom = max(total, 1)
    good_count = sum(1 for item in results if item["quality_flag"] == GOOD_FLAG)
    medium_count = sum(1 for item in results if item["quality_flag"] == MEDIUM_FLAG)
    poor_count = sum(1 for item in results if item["quality_flag"] == POOR_FLAG)
    good_rate = good_count / denom
    usable_rate = (good_count + medium_count) / denom
    avg_conf = float(np.mean([item["avg_confidence"] for item in results])) if results else 0.0
    avg_cov = float(np.mean([item["text_coverage_pct"] for item in results])) if results else 0.0
    total_lines = int(sum(item["filtered_lines"] for item in results))

    if good_rate >= 0.70:
        decision = "PROCEED"
        verdict = "PROCEED - enough images passed the GOOD gate for COCO bootstrapping."
    elif usable_rate >= 0.50:
        decision = "PREPROCESS"
        verdict = "PREPROCESS - inspect failures, lower --conf, or adjust deskew/denoise."
    else:
        decision = "SWITCH"
        verdict = "SWITCH - run Plan B and compare PaddleOCR/DocTR/EasyOCR outputs."

    print("\nSURYA DETECTION REPORT")
    print("=" * 96)
    print(f"{'name':36} | {'lines':>5} | {'avg_conf':>8} | {'coverage':>8} | flag")
    print("-" * 96)
    for item in results:
        name = str(item["image"])[:36]
        print(
            f"{name:36} | {item['filtered_lines']:5d} | "
            f"{item['avg_confidence']:8.4f} | {item['text_coverage_pct']:7.2f}% | "
            f"{item['quality_flag']}"
        )

    print("-" * 96)
    print(f"GOOD:   {good_count:4d}/{total:4d} ({good_count / denom * 100:5.1f}%)")
    print(f"MEDIUM: {medium_count:4d}/{total:4d} ({medium_count / denom * 100:5.1f}%)")
    print(f"POOR:   {poor_count:4d}/{total:4d} ({poor_count / denom * 100:5.1f}%)")
    print(f"Average confidence: {avg_conf:.4f}")
    print(f"Average coverage:   {avg_cov:.2f}%")
    print(f"Total text lines:   {total_lines}")
    print(f"DECISION GATE:      {verdict}")
    print("=" * 96)

    return {
        "total": total,
        "good_count": good_count,
        "medium_count": medium_count,
        "poor_count": poor_count,
        "good_rate": round(good_rate, 4),
        "usable_rate": round(usable_rate, 4),
        "avg_confidence": round(avg_conf, 4),
        "avg_coverage_pct": round(avg_cov, 2),
        "total_lines": total_lines,
        "decision": decision,
        "verdict": verdict,
    }


def save_report(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    out_path: Path | None = None,
) -> Path:
    """Save the summary and per-image evaluation report as JSON."""
    if out_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = config.LOG_DIR / f"report_{timestamp}.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": summary,
        "per_image": results,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    logger.info("Saved evaluation report to %s", out_path)
    return out_path


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU for two [x1, y1, x2, y2] boxes."""
    x_left = max(float(box1[0]), float(box2[0]))
    y_top = max(float(box1[1]), float(box2[1]))
    x_right = min(float(box1[2]), float(box2[2]))
    y_bottom = min(float(box1[3]), float(box2[3]))

    intersection = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def evaluate_with_ground_truth(
    gt_boxes: list[list[float]],
    pred_boxes: list[list[float]],
    iou_thresh: float = 0.5,
) -> dict[str, float | int]:
    """Evaluate predicted boxes against ground-truth boxes using greedy IoU matching."""
    matched_gt: set[int] = set()
    true_positive = 0
    false_positive = 0

    for pred_box in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = gt_idx

        if best_iou >= iou_thresh and best_idx >= 0:
            true_positive += 1
            matched_gt.add(best_idx)
        else:
            false_positive += 1

    false_negative = len(gt_boxes) - len(matched_gt)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1_score = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)

    return {
        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1": round(float(f1_score), 4),
    }
