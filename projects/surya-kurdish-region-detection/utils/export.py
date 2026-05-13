"""COCO and per-image JSON exporters for Surya annotation outputs."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

try:
    from .. import config
except ImportError:  # pragma: no cover - used when scripts run from this directory.
    import config  # type: ignore

logger = logging.getLogger(__name__)


def _prediction_items(prediction: Any, key: str = "bboxes") -> list[Any]:
    """Return bbox-like items from object-style or dict-style Surya output."""
    if prediction is None:
        return []
    if isinstance(prediction, dict):
        value = prediction.get(key)
        if value is None and key == "bboxes":
            value = prediction.get("text_lines") or prediction.get("layout") or []
        return list(value or [])
    return list(getattr(prediction, key, []) or [])


def _item_bbox(item: Any) -> list[float]:
    """Extract a Surya-format [x1, y1, x2, y2] bbox from a prediction item."""
    if isinstance(item, dict):
        bbox = item.get("bbox") or item.get("box") or item.get("rectangle") or [0, 0, 0, 0]
    else:
        bbox = getattr(item, "bbox", [0, 0, 0, 0])
    return [float(v) for v in list(bbox)[:4]]


def _item_confidence(item: Any) -> float:
    """Extract confidence from a detection item."""
    if isinstance(item, dict):
        value = item.get("confidence", item.get("score", 0.0))
    else:
        value = getattr(item, "confidence", getattr(item, "score", 0.0))
    return float(value or 0.0)


def _item_score(item: Any) -> float:
    """Extract score from a layout item."""
    if isinstance(item, dict):
        value = item.get("score", item.get("confidence", 0.0))
    else:
        value = getattr(item, "score", getattr(item, "confidence", 0.0))
    return float(value or 0.0)


def _item_label(item: Any) -> str:
    """Extract a layout label from a prediction item."""
    if isinstance(item, dict):
        return str(item.get("label", item.get("name", "unknown")) or "unknown")
    return str(getattr(item, "label", "unknown") or "unknown")


def bbox_surya_to_coco(bbox: Sequence[float]) -> list[float]:
    """Convert Surya [x1, y1, x2, y2] to rounded COCO [x, y, width, height]."""
    x1, y1, x2, y2 = [float(v) for v in list(bbox)[:4]]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return [round(x1, 1), round(y1, 1), round(width, 1), round(height, 1)]


def build_coco(
    image_data: list[dict[str, Any]],
    det_preds: list[Any],
    lay_preds: list[Any],
    conf_thresh: float = config.CONF_THRESH,
    dataset_name: str = "Kurdish Sorani Surya Text Regions",
) -> dict[str, Any]:
    """Build a full COCO annotation dictionary from Surya text-line and layout predictions."""
    created_at = datetime.now().isoformat(timespec="seconds")
    coco: dict[str, Any] = {
        "info": {
            "description": dataset_name,
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Surya Kurdish local auto-annotation pipeline",
            "date_created": created_at,
        },
        "licenses": [
            {
                "id": 0,
                "name": "Unknown or user-provided",
                "url": "",
            }
        ],
        "categories": config.COCO_CATEGORIES,
        "images": [],
        "annotations": [],
    }

    annotation_id = 1
    for image_id, (image_info, det_pred, lay_pred) in enumerate(zip(image_data, det_preds, lay_preds), start=1):
        file_name = str(image_info.get("file_name", image_info.get("name", image_info.get("path", ""))))
        width = int(image_info["width"])
        height = int(image_info["height"])
        coco["images"].append(
            {
                "id": image_id,
                "file_name": Path(file_name).name,
                "width": width,
                "height": height,
            }
        )

        for det_item in _prediction_items(det_pred):
            confidence = _item_confidence(det_item)
            if confidence < conf_thresh:
                continue
            coco_bbox = bbox_surya_to_coco(_item_bbox(det_item))
            area = round(coco_bbox[2] * coco_bbox[3], 1)
            if area <= 0.0:
                continue
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": coco_bbox,
                    "area": area,
                    "segmentation": [],
                    "iscrowd": 0,
                    "attributes": {
                        "confidence": round(confidence, 4),
                        "source": "surya_detection",
                    },
                }
            )
            annotation_id += 1

        for layout_item in _prediction_items(lay_pred):
            label = _item_label(layout_item)
            score = _item_score(layout_item)
            category_id = int(config.LAYOUT_LABEL_TO_COCO_ID.get(label, 2))
            coco_bbox = bbox_surya_to_coco(_item_bbox(layout_item))
            area = round(coco_bbox[2] * coco_bbox[3], 1)
            if area <= 0.0:
                continue
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": coco_bbox,
                    "area": area,
                    "segmentation": [],
                    "iscrowd": 0,
                    "attributes": {
                        "score": round(score, 4),
                        "surya_label": label,
                        "source": "surya_layout",
                    },
                }
            )
            annotation_id += 1

    return coco


def save_coco(coco: dict[str, Any], out_path: Path | None = None) -> Path:
    """Save a COCO annotation dictionary to a timestamped JSON file if no path is given."""
    if out_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = config.ANN_DIR / f"coco_annotations_{timestamp}.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(coco, handle, ensure_ascii=False, indent=2)
    logger.info("Saved COCO annotations to %s", out_path)
    return out_path


def export_per_image_json(
    image_data: list[dict[str, Any]],
    det_preds: list[Any],
    lay_preds: list[Any],
    out_dir: Path,
    conf_thresh: float = config.CONF_THRESH,
) -> list[Path]:
    """Export one compact JSON file per image with COCO-format bboxes for inspection."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for image_info, det_pred, lay_pred in zip(image_data, det_preds, lay_preds):
        file_name = str(image_info.get("file_name", image_info.get("name", image_info.get("path", ""))))
        payload = {
            "file": Path(file_name).name,
            "width": int(image_info["width"]),
            "height": int(image_info["height"]),
            "bbox_format": "coco_xywh",
            "text_lines": [
                {
                    "bbox": bbox_surya_to_coco(_item_bbox(det_item)),
                    "confidence": round(_item_confidence(det_item), 4),
                }
                for det_item in _prediction_items(det_pred)
                if _item_confidence(det_item) >= conf_thresh
            ],
            "layout": [
                {
                    "bbox": bbox_surya_to_coco(_item_bbox(layout_item)),
                    "label": _item_label(layout_item),
                    "score": round(_item_score(layout_item), 4),
                }
                for layout_item in _prediction_items(lay_pred)
            ],
        }
        out_path = out_dir / f"{Path(file_name).stem}.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        saved_paths.append(out_path)

    logger.info("Saved %d per-image JSON files to %s", len(saved_paths), out_dir)
    return saved_paths
