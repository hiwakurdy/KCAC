"""Fallback OCR detection engines for building COCO text-line annotations."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback before requirements install.
    def tqdm(iterable: Any, **_: Any) -> Any:
        """Return the iterable unchanged when tqdm is not installed."""
        return iterable

try:
    from . import config
except ImportError:  # pragma: no cover - used when run as python plan_b.py.
    import config  # type: ignore

logger = logging.getLogger(__name__)


class PaddleEngine:
    """PaddleOCR detector configured for Arabic-script text."""

    def __init__(self) -> None:
        """Import and initialize PaddleOCR with GPU text detection settings."""
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            print("Install PaddleOCR first: pip install paddlepaddle paddleocr")
            raise SystemExit(1) from exc

        self.ocr = PaddleOCR(
            lang="ar",
            use_gpu=True,
            use_angle_cls=True,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            show_log=False,
        )

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        """Detect text boxes and return axis-aligned [x1, y1, x2, y2] boxes."""
        results = self.ocr.ocr(np.array(image.convert("RGB")), cls=True)
        detections: list[dict[str, Any]] = []
        if not results:
            return detections

        lines = results[0] if isinstance(results, list) and results and isinstance(results[0], list) else results
        for line in lines or []:
            parsed = _parse_paddle_line(line)
            if parsed is not None:
                detections.append(parsed)
        return detections


class DocTREngine:
    """DocTR detector using DB-ResNet50 on CUDA."""

    def __init__(self) -> None:
        """Import and initialize DocTR OCR predictor on CUDA."""
        try:
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor
        except ImportError as exc:
            print("Install DocTR first: pip install python-doctr[torch]")
            raise SystemExit(1) from exc

        self.document_file = DocumentFile
        self.model = ocr_predictor(det_arch="db_resnet50", pretrained=True).cuda()

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        """Detect DocTR line boxes, converting relative coordinates into image pixels."""
        img_np = np.array(image.convert("RGB"))
        doc = self.document_file.from_images([img_np])
        result = self.model(doc)
        width, height = image.size
        detections: list[dict[str, Any]] = []

        if not result.pages:
            return detections

        for block in result.pages[0].blocks:
            for line in block.lines:
                geometry = getattr(line, "geometry", None)
                if geometry is None and getattr(line, "words", None):
                    geometry = _union_relative_word_geometry(line.words)
                if geometry is None:
                    continue
                (x1n, y1n), (x2n, y2n) = geometry
                words = getattr(line, "words", [])
                confidence = float(np.mean([getattr(word, "confidence", 1.0) for word in words])) if words else 1.0
                text = " ".join(getattr(word, "value", "") for word in words).strip()
                detections.append(
                    {
                        "bbox": [x1n * width, y1n * height, x2n * width, y2n * height],
                        "confidence": confidence,
                        "text": text,
                    }
                )
        return detections


class EasyOCREngine:
    """EasyOCR detector for Arabic-script text."""

    def __init__(self) -> None:
        """Import and initialize EasyOCR with Arabic language support on GPU."""
        try:
            import easyocr
        except ImportError as exc:
            print("Install EasyOCR first: pip install easyocr")
            raise SystemExit(1) from exc

        self.reader = easyocr.Reader(["ar"], gpu=True)

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        """Detect EasyOCR boxes with reader.detect, assigning confidence 1.0 to detector boxes."""
        result = self.reader.detect(np.array(image.convert("RGB")))
        boxes = _extract_easyocr_boxes(result)
        return [{"bbox": box, "confidence": 1.0, "text": ""} for box in boxes]


def _parse_paddle_line(line: Any) -> dict[str, Any] | None:
    """Parse a PaddleOCR output row into a normalized detection dictionary."""
    if isinstance(line, dict):
        points = line.get("dt_polys") or line.get("points") or line.get("bbox")
        confidence = float(line.get("rec_score", line.get("confidence", 1.0)))
        text = str(line.get("rec_text", line.get("text", "")))
    elif isinstance(line, (list, tuple)) and len(line) >= 2:
        points = line[0]
        rec = line[1]
        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
            text = str(rec[0])
            confidence = float(rec[1])
        else:
            text = ""
            confidence = 1.0
    else:
        return None

    bbox = _polygon_to_bbox(points)
    if bbox is None:
        return None
    return {"bbox": bbox, "confidence": confidence, "text": text}


def _polygon_to_bbox(points: Any) -> list[float] | None:
    """Convert a polygon-like point list to an axis-aligned [x1, y1, x2, y2] bbox."""
    if points is None:
        return None
    arr = np.asarray(points, dtype=float)
    if arr.size < 4:
        return None
    arr = arr.reshape(-1, 2)
    xs = arr[:, 0]
    ys = arr[:, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _union_relative_word_geometry(words: list[Any]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Union DocTR word geometries in relative coordinates."""
    geometries = [getattr(word, "geometry", None) for word in words]
    geometries = [geometry for geometry in geometries if geometry is not None]
    if not geometries:
        return None
    x1 = min(float(geometry[0][0]) for geometry in geometries)
    y1 = min(float(geometry[0][1]) for geometry in geometries)
    x2 = max(float(geometry[1][0]) for geometry in geometries)
    y2 = max(float(geometry[1][1]) for geometry in geometries)
    return (x1, y1), (x2, y2)


def _extract_easyocr_boxes(result: Any) -> list[list[float]]:
    """Extract and normalize EasyOCR horizontal detector boxes from nested detect output."""
    boxes: list[list[float]] = []

    def visit(node: Any) -> None:
        """Recursively collect [x_min, x_max, y_min, y_max] EasyOCR boxes."""
        if isinstance(node, np.ndarray):
            visit(node.tolist())
            return
        if isinstance(node, (list, tuple)):
            if len(node) == 4 and all(isinstance(value, (int, float, np.integer, np.floating)) for value in node):
                x1, x2, y1, y2 = [float(value) for value in node]
                boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                return
            for child in node:
                visit(child)

    visit(result)
    return boxes


def collect_image_paths(image_dir: Path, sample: int | None = None) -> list[Path]:
    """Collect sorted images from an input directory."""
    image_dir = Path(image_dir)
    paths = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in config.IMG_EXTS)
    if sample is not None:
        paths = paths[: max(0, sample)]
    if not paths:
        logger.error("No images found in %s", image_dir)
        raise SystemExit(1)
    return paths


def bbox_surya_to_coco(bbox: list[float]) -> list[float]:
    """Convert [x1, y1, x2, y2] to rounded COCO [x, y, width, height]."""
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return [round(x1, 1), round(y1, 1), round(max(0.0, x2 - x1), 1), round(max(0.0, y2 - y1), 1)]


def convert_to_coco_format(
    image_paths: list[Path],
    all_detections: list[list[dict[str, Any]]],
    conf_thresh: float = 0.35,
) -> dict[str, Any]:
    """Convert Plan B detector outputs to a COCO dictionary using category_id=1 for text_line."""
    coco: dict[str, Any] = {
        "info": {
            "description": "Kurdish Sorani Plan B text-line annotations",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(timespec="seconds"),
        },
        "licenses": [{"id": 0, "name": "Unknown or user-provided", "url": ""}],
        "categories": [{"id": 1, "name": "text_line", "supercategory": "text"}],
        "images": [],
        "annotations": [],
    }
    annotation_id = 1

    for image_id, (image_path, detections) in enumerate(zip(image_paths, all_detections), start=1):
        with Image.open(image_path) as img:
            width, height = img.size
        coco["images"].append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})

        for detection in detections:
            confidence = float(detection.get("confidence", 0.0))
            if confidence < conf_thresh:
                continue
            coco_bbox = bbox_surya_to_coco([float(value) for value in detection["bbox"]])
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
                        "text": str(detection.get("text", "")),
                        "source": "plan_b",
                    },
                }
            )
            annotation_id += 1
    return coco


def save_viz_planb(image: Image.Image, dets: list[dict[str, Any]], out_path: Path, conf_thresh: float) -> None:
    """Save a simple OpenCV visualization with green boxes for accepted detections."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    color = (0, 220, 80)
    for detection in dets:
        confidence = float(detection.get("confidence", 0.0))
        if confidence < conf_thresh:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{confidence:.2f}", (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    cv2.imwrite(str(out_path), canvas)


def parse_args() -> argparse.Namespace:
    """Parse Plan B command-line options."""
    parser = argparse.ArgumentParser(description="Plan B OCR text-line detection pipeline")
    parser.add_argument("--engine", choices=["paddle", "doctr", "easyocr"], default="paddle")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--image-dir", type=Path, default=config.IMAGE_DIR)
    return parser.parse_args()


def configure_logging() -> None:
    """Configure simple console and file logging for Plan B runs."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.LOG_DIR / "plan_b.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
    )


def load_engine(engine_name: str) -> PaddleEngine | DocTREngine | EasyOCREngine:
    """Instantiate the selected Plan B engine."""
    engines: dict[str, type[PaddleEngine] | type[DocTREngine] | type[EasyOCREngine]] = {
        "paddle": PaddleEngine,
        "doctr": DocTREngine,
        "easyocr": EasyOCREngine,
    }
    return engines[engine_name]()


def main() -> None:
    """Run the selected Plan B engine over images and export COCO annotations."""
    args = parse_args()
    configure_logging()
    paths = collect_image_paths(args.image_dir, args.sample)
    logger.info("Running Plan B with %s on %d images", args.engine, len(paths))
    engine = load_engine(args.engine)

    all_detections: list[list[dict[str, Any]]] = []
    viz_dir = config.VIZ_DIR / f"plan_b_{args.engine}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    accepted_total = 0
    for path in tqdm(paths, desc=f"Detecting with {args.engine}", unit="image"):
        with Image.open(path) as raw_image:
            image = raw_image.convert("RGB")
        start = time.time()
        detections = engine.detect(image)
        elapsed = time.time() - start
        accepted = sum(1 for detection in detections if float(detection.get("confidence", 0.0)) >= args.conf)
        accepted_total += accepted
        all_detections.append(detections)
        save_viz_planb(image, detections, viz_dir / f"{path.stem}_planb.jpg", args.conf)
        logger.info("%s | %d accepted boxes | %.2fs", path.name, accepted, elapsed)

    coco = convert_to_coco_format(paths, all_detections, args.conf)
    out_path = config.ANN_DIR / f"plan_b_{args.engine}_coco_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(coco, handle, ensure_ascii=False, indent=2)

    print("\nPLAN B SUMMARY")
    print("=" * 72)
    print(f"Engine:       {args.engine}")
    print(f"Images:       {len(paths)}")
    print(f"Annotations:  {accepted_total}")
    print(f"COCO JSON:    {out_path}")
    print(f"Visuals:      {viz_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
