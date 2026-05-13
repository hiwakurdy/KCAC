"""Central configuration for the Kurdish Sorani Surya annotation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - handled at runtime before model loading.
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = ROOT_DIR / "images"
OUTPUT_DIR = ROOT_DIR / "results"
VIZ_DIR = OUTPUT_DIR / "visualizations"
ANN_DIR = OUTPUT_DIR / "annotations"
LOG_DIR = OUTPUT_DIR / "logs"

for directory in (IMAGE_DIR, OUTPUT_DIR, VIZ_DIR, ANN_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
CONF_THRESH = 0.40
SAMPLE_SIZE = None
SAVE_VIZ = True
EXPORT_COCO = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

GATE_GOOD = 0.70
GATE_MEDIUM = 0.50
MIN_LINES = 3
MIN_COV_PCT = 5.0

COCO_CATEGORIES = [
    {"id": 1, "name": "text_line", "supercategory": "text"},
    {"id": 2, "name": "paragraph", "supercategory": "layout"},
    {"id": 3, "name": "title", "supercategory": "layout"},
    {"id": 4, "name": "table", "supercategory": "layout"},
    {"id": 5, "name": "figure", "supercategory": "layout"},
    {"id": 6, "name": "caption", "supercategory": "layout"},
    {"id": 7, "name": "list_item", "supercategory": "layout"},
    {"id": 8, "name": "page_header", "supercategory": "layout"},
    {"id": 9, "name": "page_footer", "supercategory": "layout"},
]

LAYOUT_LABEL_TO_COCO_ID = {
    "Text": 2,
    "Text-inline-math": 2,
    "Paragraph": 2,
    "Title": 3,
    "Section-header": 3,
    "SectionHeader": 3,
    "Heading": 3,
    "Table": 4,
    "Figure": 5,
    "Picture": 5,
    "Image": 5,
    "Caption": 6,
    "Footnote": 6,
    "List-item": 7,
    "ListItem": 7,
    "List": 7,
    "Page-header": 8,
    "PageHeader": 8,
    "Page-footer": 9,
    "PageFooter": 9,
}

VIZ_COLORS = {
    "text_line": (0, 220, 80),
    "Text": (40, 170, 80),
    "Text-inline-math": (40, 150, 110),
    "Paragraph": (40, 170, 80),
    "Title": (40, 70, 230),
    "Section-header": (0, 150, 230),
    "SectionHeader": (0, 150, 230),
    "Heading": (0, 150, 230),
    "Table": (220, 90, 60),
    "Figure": (190, 50, 190),
    "Picture": (190, 50, 190),
    "Image": (190, 50, 190),
    "Caption": (210, 180, 35),
    "Footnote": (190, 165, 55),
    "List-item": (0, 190, 210),
    "ListItem": (0, 190, 210),
    "List": (0, 190, 210),
    "Page-header": (115, 115, 115),
    "PageHeader": (115, 115, 115),
    "Page-footer": (90, 90, 90),
    "PageFooter": (90, 90, 90),
    "unknown": (170, 170, 170),
}
