"""Shared model and dataset helpers for local Surya-style region detection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

try:
    from .. import config
except ImportError:  # pragma: no cover - used when scripts run from this directory.
    import config  # type: ignore

logger = logging.getLogger(__name__)

NUM_REGION_CLASSES = 10
CATEGORY_ID_TO_NAME = {
    0: "__background__",
    1: "text_line",
    2: "paragraph",
    3: "title",
    4: "table",
    5: "figure",
    6: "caption",
    7: "list_item",
    8: "page_header",
    9: "page_footer",
}
CATEGORY_NAME_TO_ID = {name: idx for idx, name in CATEGORY_ID_TO_NAME.items()}
LABEL_TO_CATEGORY_ID = {
    **CATEGORY_NAME_TO_ID,
    **config.LAYOUT_LABEL_TO_COCO_ID,
    "Picture": 5,
    "Text": 2,
    "Title": 3,
    "Table": 4,
    "Figure": 5,
    "Caption": 6,
    "ListItem": 7,
    "PageHeader": 8,
    "PageFooter": 9,
}


def collect_image_paths(image_dir: Path) -> list[Path]:
    """Collect supported images from a directory in stable sorted order."""
    image_dir = Path(image_dir)
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in config.IMG_EXTS)


def label_to_category_id(label: str) -> int:
    """Map a Surya or COCO label name to a stable local category ID."""
    label = str(label)
    if label in LABEL_TO_CATEGORY_ID:
        return int(LABEL_TO_CATEGORY_ID[label])
    normalized = label.lower().replace("-", "_").replace(" ", "_")
    return int(CATEGORY_NAME_TO_ID.get(normalized, 2))


def coco_xywh_to_xyxy(bbox: list[float]) -> list[float]:
    """Convert COCO [x, y, width, height] into [x1, y1, x2, y2]."""
    x, y, width, height = [float(value) for value in bbox[:4]]
    return [x, y, x + max(0.0, width), y + max(0.0, height)]


def clamp_box(box: list[float], width: int, height: int) -> list[float] | None:
    """Clamp a box to image boundaries and return None if it becomes invalid."""
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(x1, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_xyxy_to_coco(box: list[float]) -> list[float]:
    """Convert [x1, y1, x2, y2] to rounded COCO [x, y, width, height]."""
    x1, y1, x2, y2 = [float(value) for value in box]
    return [round(x1, 1), round(y1, 1), round(max(0.0, x2 - x1), 1), round(max(0.0, y2 - y1), 1)]


def load_per_image_annotation(
    anno_path: Path,
    width: int,
    height: int,
    include_text_lines: bool = True,
    include_layout: bool = True,
    min_conf: float = 0.0,
    min_score: float = 0.0,
) -> tuple[list[list[float]], list[int]]:
    """Load one per-image JSON annotation file and return boxes plus labels."""
    with Path(anno_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    boxes: list[list[float]] = []
    labels: list[int] = []
    bbox_format = str(data.get("bbox_format", "coco_xywh")).lower()

    def normalize_bbox(raw_bbox: list[float]) -> list[float] | None:
        """Normalize one bbox from the annotation format into clamped xyxy coordinates."""
        box = coco_xywh_to_xyxy(raw_bbox) if bbox_format in {"coco_xywh", "xywh", "coco"} else raw_bbox
        return clamp_box(box, width, height)

    if include_text_lines:
        for item in data.get("text_lines", []):
            confidence = float(item.get("confidence", 1.0))
            if confidence < min_conf:
                continue
            box = normalize_bbox(item["bbox"])
            if box is None:
                continue
            boxes.append(box)
            labels.append(1)

    if include_layout:
        for item in data.get("layout", []):
            score = float(item.get("score", item.get("confidence", 1.0)))
            if score < min_score:
                continue
            box = normalize_bbox(item["bbox"])
            if box is None:
                continue
            boxes.append(box)
            labels.append(label_to_category_id(str(item.get("label", "Text"))))

    return boxes, labels


class PerImageJsonRegionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Torch dataset for images with the per-image JSON emitted by this Surya pipeline."""

    def __init__(
        self,
        image_dir: Path,
        anno_dir: Path,
        include_text_lines: bool = True,
        include_layout: bool = True,
        min_conf: float = 0.0,
        min_score: float = 0.0,
        max_images: int | None = None,
    ) -> None:
        """Create a dataset from matching image and JSON basenames."""
        self.image_dir = Path(image_dir)
        self.anno_dir = Path(anno_dir)
        self.include_text_lines = include_text_lines
        self.include_layout = include_layout
        self.min_conf = float(min_conf)
        self.min_score = float(min_score)
        image_paths = collect_image_paths(self.image_dir)
        self.records: list[tuple[Path, Path]] = []
        for image_path in image_paths:
            anno_path = self.anno_dir / f"{image_path.stem}.json"
            if anno_path.exists():
                self.records.append((image_path, anno_path))
        if max_images is not None:
            self.records = self.records[: max(0, int(max_images))]
        if not self.records:
            raise ValueError(f"No matching image/json pairs found in {self.image_dir} and {self.anno_dir}")
        logger.info("Loaded %d image/json pairs from %s", len(self.records), self.image_dir)

    def __len__(self) -> int:
        """Return the number of matched image/json records."""
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return one image tensor and Faster R-CNN target dictionary."""
        image_path, anno_path = self.records[index]
        with Image.open(image_path) as raw_image:
            image = ImageOps.exif_transpose(raw_image).convert("RGB")
        width, height = image.size
        boxes, labels = load_per_image_annotation(
            anno_path,
            width,
            height,
            include_text_lines=self.include_text_lines,
            include_layout=self.include_layout,
            min_conf=self.min_conf,
            min_score=self.min_score,
        )
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        if boxes_tensor.numel() == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]).clamp(min=0) * (boxes_tensor[:, 3] - boxes_tensor[:, 1]).clamp(min=0)
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((boxes_tensor.shape[0],), dtype=torch.int64),
        }
        return F.to_tensor(image), target

    def image_path(self, index: int) -> Path:
        """Return the image path for a dataset index."""
        return self.records[index][0]


def collate_detection_batch(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Collate variable-size detection samples into lists for torchvision detectors."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_region_detector(
    num_classes: int = NUM_REGION_CLASSES,
    pretrained: bool = True,
    min_size: int = 800,
    max_size: int = 1333,
) -> torch.nn.Module:
    """Build a Faster R-CNN detector with a Surya-region classification head."""
    weights = None
    weights_backbone = None
    if pretrained:
        try:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        except Exception as exc:  # pragma: no cover - depends on torchvision registry.
            logger.warning("Could not select pretrained Faster R-CNN weights: %s", exc)

    try:
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=weights_backbone,
            min_size=min_size,
            max_size=max_size,
        )
    except Exception as exc:
        if not pretrained:
            raise
        logger.warning("Pretrained model load failed; retrying without pretrained weights: %s", exc)
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            min_size=min_size,
            max_size=max_size,
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def move_targets_to_device(
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Move a list of target dictionaries to the requested device."""
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_f1: float,
    args: dict[str, Any],
) -> Path:
    """Save a model checkpoint that can be loaded later for prediction or resumed training."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "best_f1": float(best_f1),
        "num_classes": NUM_REGION_CLASSES,
        "category_id_to_name": CATEGORY_ID_TO_NAME,
        "args": args,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
    min_size: int = 800,
    max_size: int = 1333,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a saved region detector checkpoint and return model plus checkpoint payload."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    num_classes = int(checkpoint.get("num_classes", NUM_REGION_CLASSES))
    model = build_region_detector(num_classes=num_classes, pretrained=False, min_size=min_size, max_size=max_size)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint
