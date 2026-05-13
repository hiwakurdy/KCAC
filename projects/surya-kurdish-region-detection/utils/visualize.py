"""Visualization helpers for Surya text-line and layout predictions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
    """Extract a [x1, y1, x2, y2] bbox from a prediction item."""
    if isinstance(item, dict):
        bbox = item.get("bbox") or item.get("box") or item.get("rectangle") or [0, 0, 0, 0]
    else:
        bbox = getattr(item, "bbox", [0, 0, 0, 0])
    return [float(v) for v in list(bbox)[:4]]


def _item_confidence(item: Any) -> float:
    """Extract confidence from a prediction item."""
    if isinstance(item, dict):
        value = item.get("confidence", item.get("score", 0.0))
    else:
        value = getattr(item, "confidence", getattr(item, "score", 0.0))
    return float(value or 0.0)


def _item_score(item: Any) -> float:
    """Extract layout score from a prediction item."""
    if isinstance(item, dict):
        value = item.get("score", item.get("confidence", 0.0))
    else:
        value = getattr(item, "score", getattr(item, "confidence", 0.0))
    return float(value or 0.0)


def _item_label(item: Any) -> str:
    """Extract layout label from a prediction item."""
    if isinstance(item, dict):
        return str(item.get("label", item.get("name", "unknown")) or "unknown")
    return str(getattr(item, "label", "unknown") or "unknown")


def _rgb_from_bgr(color: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert a BGR OpenCV color tuple to a matplotlib RGB float tuple."""
    b, g, r = color
    return (r / 255.0, g / 255.0, b / 255.0)


def draw_detections_cv2(
    image: Image.Image,
    det_pred: Any,
    lay_pred: Any,
    conf_thresh: float = config.CONF_THRESH,
) -> np.ndarray:
    """Draw layout regions and text-line boxes on an image and return a BGR OpenCV array."""
    canvas = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    overlay = canvas.copy()

    for layout_item in _prediction_items(lay_pred):
        x1, y1, x2, y2 = [int(round(v)) for v in _item_bbox(layout_item)]
        label = _item_label(layout_item)
        color = config.VIZ_COLORS.get(label, config.VIZ_COLORS["unknown"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)

    canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)

    for layout_item in _prediction_items(lay_pred):
        x1, y1, x2, y2 = [int(round(v)) for v in _item_bbox(layout_item)]
        label = _item_label(layout_item)
        score = _item_score(layout_item)
        color = config.VIZ_COLORS.get(label, config.VIZ_COLORS["unknown"])
        text = f"{label} {score:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=2)
        text_y = max(14, y1 - 4)
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, text_y - text_h - 4), (x1 + text_w + 6, text_y + 2), color, -1)
        cv2.putText(canvas, text, (x1 + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    line_color = config.VIZ_COLORS["text_line"]
    for det_item in _prediction_items(det_pred):
        confidence = _item_confidence(det_item)
        if confidence < conf_thresh:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in _item_bbox(det_item)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), line_color, thickness=1)
        cv2.putText(
            canvas,
            f"{confidence:.2f}",
            (x1, max(10, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            line_color,
            1,
            cv2.LINE_AA,
        )

    return canvas


def save_panel_viz(
    image: Image.Image,
    det_pred: Any,
    lay_pred: Any,
    out_path: Path,
    title: str,
    quality_flag: str,
    conf_thresh: float = config.CONF_THRESH,
) -> None:
    """Save a JPG panel with original image, text-line detections, and layout regions."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_image = np.array(image.convert("RGB"))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=110)
    flag_color = "#1f8f55" if "GOOD" in quality_flag else "#c78416" if "MEDIUM" in quality_flag else "#c43d3d"
    fig.suptitle(f"{title} | {quality_flag}", color=flag_color, fontsize=13, fontweight="bold")

    axes[0].imshow(rgb_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(rgb_image)
    text_line_count = 0
    for det_item in _prediction_items(det_pred):
        confidence = _item_confidence(det_item)
        if confidence < conf_thresh:
            continue
        text_line_count += 1
        x1, y1, x2, y2 = _item_bbox(det_item)
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1.2,
            edgecolor="lime",
            facecolor="lime",
            alpha=0.20,
        )
        axes[1].add_patch(rect)
        axes[1].text(x1, max(0, y1 - 3), f"{confidence:.2f}", color="lime", fontsize=6, weight="bold")
    axes[1].set_title(f"Text Lines ({text_line_count}, conf >= {conf_thresh:.2f})")
    axes[1].axis("off")

    axes[2].imshow(rgb_image)
    layout_counts: dict[str, int] = {}
    for layout_item in _prediction_items(lay_pred):
        label = _item_label(layout_item)
        score = _item_score(layout_item)
        layout_counts[label] = layout_counts.get(label, 0) + 1
        color = _rgb_from_bgr(config.VIZ_COLORS.get(label, config.VIZ_COLORS["unknown"]))
        x1, y1, x2, y2 = _item_bbox(layout_item)
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1.8,
            edgecolor=color,
            facecolor=color,
            alpha=0.25,
        )
        axes[2].add_patch(rect)
        axes[2].text(
            x1 + 3,
            y1 + 12,
            f"{label} {score:.2f}",
            color=color,
            fontsize=6.5,
            weight="bold",
            bbox={"facecolor": "black", "alpha": 0.72, "pad": 1.5, "edgecolor": "none"},
        )
    layout_summary = ", ".join(f"{key}:{value}" for key, value in sorted(layout_counts.items())) or "none"
    axes[2].set_title(f"Layout Regions ({layout_summary})")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, format="jpg", bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved visualization to %s", out_path)
