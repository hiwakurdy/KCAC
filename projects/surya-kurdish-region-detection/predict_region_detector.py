"""Run a saved Surya-style region detector checkpoint on new images."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as F

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback before requirements install.
    def tqdm(iterable: Any, **_: Any) -> Any:
        """Return the iterable unchanged when tqdm is unavailable."""
        return iterable

try:
    from . import config
    from .utils.region_model import (
        CATEGORY_ID_TO_NAME,
        box_xyxy_to_coco,
        collect_image_paths,
        load_checkpoint_model,
    )
except ImportError:  # pragma: no cover - used when run as python predict_region_detector.py.
    import config  # type: ignore
    from utils.region_model import (  # type: ignore
        CATEGORY_ID_TO_NAME,
        box_xyxy_to_coco,
        collect_image_paths,
        load_checkpoint_model,
    )

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line prediction options."""
    parser = argparse.ArgumentParser(description="Predict text/layout regions with a trained checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results") / "trained_predictions")
    parser.add_argument("--score", type=float, default=0.35)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--per-image-json", action="store_true")
    return parser.parse_args()


def setup_logging(out_dir: Path) -> Path:
    """Configure stream and file logging for prediction."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "predict.log"
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


def load_image_tensor(path: Path) -> tuple[torch.Tensor, Image.Image]:
    """Load an image as both a Torch tensor and RGB PIL image."""
    with Image.open(path) as raw_image:
        image = ImageOps.exif_transpose(raw_image).convert("RGB")
    return F.to_tensor(image), image


def color_for_label(label_id: int) -> tuple[int, int, int]:
    """Return a BGR visualization color for a category ID."""
    name = CATEGORY_ID_TO_NAME.get(int(label_id), "unknown")
    return config.VIZ_COLORS.get(name, config.VIZ_COLORS.get("unknown", (170, 170, 170)))


def draw_prediction_viz(
    image: Image.Image,
    detections: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Draw predicted boxes with labels and scores onto a JPG visualization."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for det in detections:
        x, y, width, height = det["bbox"]
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + width))
        y2 = int(round(y + height))
        label_id = int(det["category_id"])
        color = color_for_label(label_id)
        text = f"{CATEGORY_ID_TO_NAME.get(label_id, label_id)} {det['score']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text_y = max(14, y1 - 4)
        cv2.putText(canvas, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    score_thresh: float,
) -> list[dict[str, Any]]:
    """Run model inference for one image tensor and return COCO-style detections."""
    output = model([image_tensor.to(device)])[0]
    boxes = output["boxes"].detach().cpu()
    scores = output["scores"].detach().cpu()
    labels = output["labels"].detach().cpu()
    detections: list[dict[str, Any]] = []
    for box, score, label in zip(boxes, scores, labels):
        score_value = float(score)
        if score_value < score_thresh:
            continue
        label_id = int(label)
        detections.append(
            {
                "category_id": label_id,
                "category_name": CATEGORY_ID_TO_NAME.get(label_id, "unknown"),
                "bbox": box_xyxy_to_coco([float(value) for value in box.tolist()]),
                "score": round(score_value, 4),
            }
        )
    return detections


def build_coco(
    image_records: list[dict[str, Any]],
    all_detections: list[list[dict[str, Any]]],
    dataset_name: str,
) -> dict[str, Any]:
    """Build a COCO JSON from prediction records."""
    categories = [category for category in config.COCO_CATEGORIES]
    coco: dict[str, Any] = {
        "info": {
            "description": dataset_name,
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(timespec="seconds"),
        },
        "licenses": [{"id": 0, "name": "Unknown or user-provided", "url": ""}],
        "categories": categories,
        "images": [],
        "annotations": [],
    }
    ann_id = 1
    for image_id, (record, detections) in enumerate(zip(image_records, all_detections), start=1):
        coco["images"].append(
            {
                "id": image_id,
                "file_name": record["file_name"],
                "width": record["width"],
                "height": record["height"],
            }
        )
        for det in detections:
            bbox = det["bbox"]
            area = round(float(bbox[2]) * float(bbox[3]), 1)
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(det["category_id"]),
                    "bbox": bbox,
                    "area": area,
                    "segmentation": [],
                    "iscrowd": 0,
                    "attributes": {
                        "score": det["score"],
                        "source": "trained_region_detector",
                    },
                }
            )
            ann_id += 1
    return coco


def save_per_image_json(
    image_records: list[dict[str, Any]],
    all_detections: list[list[dict[str, Any]]],
    out_dir: Path,
) -> None:
    """Save one compact JSON prediction file per image."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for record, detections in zip(image_records, all_detections):
        payload = {
            "file": record["file_name"],
            "width": record["width"],
            "height": record["height"],
            "bbox_format": "coco_xywh",
            "detections": detections,
        }
        out_path = out_dir / f"{Path(record['file_name']).stem}.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    """Load a trained checkpoint, predict regions, and export COCO plus optional visualizations."""
    args = parse_args()
    log_path = setup_logging(args.out_dir)
    logger.info("Logging to %s", log_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    model, checkpoint = load_checkpoint_model(args.checkpoint, device, min_size=args.min_size, max_size=args.max_size)
    logger.info("Loaded checkpoint from epoch %s with best_f1 %.4f", checkpoint.get("epoch"), checkpoint.get("best_f1", 0.0))

    paths = collect_image_paths(args.image_dir)
    if not paths:
        raise SystemExit(f"No images found in {args.image_dir}")
    viz_dir = Path(args.out_dir) / "visualizations"
    image_records: list[dict[str, Any]] = []
    all_detections: list[list[dict[str, Any]]] = []

    for path in tqdm(paths, desc="Predicting", unit="image"):
        image_tensor, image = load_image_tensor(path)
        detections = predict_one(model, image_tensor, device, args.score)
        image_records.append({"file_name": path.name, "width": image.width, "height": image.height})
        all_detections.append(detections)
        if not args.no_viz:
            draw_prediction_viz(image, detections, viz_dir / f"{path.stem}_trained_pred.jpg")
        logger.info("%s | %d detections", path.name, len(detections))

    coco = build_coco(image_records, all_detections, "Trained Kurdish Sorani region detector predictions")
    coco_path = Path(args.out_dir) / f"trained_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with coco_path.open("w", encoding="utf-8") as handle:
        json.dump(coco, handle, ensure_ascii=False, indent=2)
    if args.per_image_json:
        save_per_image_json(image_records, all_detections, Path(args.out_dir) / "per_image")
    logger.info("Saved COCO predictions to %s", coco_path)


if __name__ == "__main__":
    main()
