from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from ..geometry import baseline_from_polygon
from ..models import EnginePageOutput, LineOutput, PageRef, Polygon
from .base import BaseOcrEngine, EngineError, line_id, make_output


class KrakenEngine(BaseOcrEngine):
    name = "kraken"

    def version(self) -> str:
        kraken = self.require_import("kraken")
        return str(getattr(kraken, "__version__", "kraken"))

    def detect_lines(self, image_path: Path, page_id: str) -> list[LineOutput]:
        try:
            from kraken import binarization, pageseg
        except ImportError as exc:
            raise EngineError(self.name, "Kraken is required for line detection", recoverable=False) from exc

        with Image.open(image_path) as image:
            bw = binarization.nlbin(image.convert("L"))
            segmentation = pageseg.segment(bw, text_direction="horizontal-rl")

        boxes = self._segmentation_lines(segmentation)
        lines: list[LineOutput] = []
        for index, item in enumerate(boxes, start=1):
            boundary = self._line_boundary(item)
            polygon = self._boundary_to_polygon(boundary)
            lines.append(
                LineOutput(
                    line_id=line_id(page_id, index),
                    polygon=polygon,
                    baseline=baseline_from_polygon(polygon),
                    text="",
                    confidence=None,
                    reading_order=index,
                )
            )
        return lines

    def run(self, page: PageRef) -> EnginePageOutput:
        try:
            from kraken import rpred
            from kraken.lib import models
        except ImportError as exc:
            raise EngineError(self.name, "Kraken OCR modules are required", recoverable=False) from exc

        lines = self.detect_lines(page.image_path, page.page_id)
        if not self.config.model:
            return make_output(page, self.name, f"{self.version()} line-detection-only", lines)
        model_path = Path(self.config.model)
        if not model_path.exists():
            raise EngineError(
                self.name,
                f"Kraken model checkpoint not found: {model_path}",
                recoverable=False,
            )

        with Image.open(page.image_path) as image:
            model = models.load_any(str(model_path))
            bounds = {
                "text_direction": "horizontal-rl",
                "boxes": [
                    {"boundary": [(int(x), int(y)) for x, y in line.polygon], "text_direction": "horizontal-rl"}
                    for line in lines
                ],
            }
            predictions = list(rpred.rpred(model, image, bounds))

        recognised: list[LineOutput] = []
        for index, line in enumerate(lines):
            text = ""
            confidence = None
            if index < len(predictions):
                pred = predictions[index]
                text = str(getattr(pred, "prediction", getattr(pred, "text", "")))
                cuts = getattr(pred, "cuts", None)
                if cuts:
                    confidences = [float(item[-1]) for item in cuts if item and isinstance(item[-1], float | int)]
                    if confidences:
                        confidence = sum(confidences) / len(confidences)
            recognised.append(
                LineOutput(
                    line_id=line.line_id,
                    polygon=line.polygon,
                    baseline=line.baseline,
                    text=text,
                    confidence=confidence,
                    reading_order=line.reading_order,
                )
            )
        return make_output(page, self.name, self.version(), recognised)

    @staticmethod
    def _segmentation_lines(segmentation: object) -> list[object]:
        if isinstance(segmentation, dict):
            return list(segmentation.get("boxes") or segmentation.get("lines") or [])
        for attr in ("boxes", "lines"):
            value = getattr(segmentation, attr, None)
            if value:
                return list(value)
        regions = getattr(segmentation, "regions", None)
        if not regions:
            return []
        lines: list[object] = []
        for region in regions:
            if isinstance(region, dict):
                region_lines = region.get("lines", [])
            else:
                region_lines = getattr(region, "lines", [])
            lines.extend(list(region_lines or []))
        return lines

    @staticmethod
    def _line_boundary(item: object) -> object:
        if isinstance(item, dict):
            return item.get("boundary") or item.get("bbox") or item.get("box") or item.get("polygon")
        for attr in ("boundary", "bbox", "box", "polygon"):
            value = getattr(item, attr, None)
            if value:
                return value
        return item

    @staticmethod
    def _boundary_to_polygon(boundary: object) -> Polygon:
        if not boundary:
            return []
        if isinstance(boundary, list | tuple) and len(boundary) == 4 and all(isinstance(v, int | float) for v in boundary):
            x1, y1, x2, y2 = [float(v) for v in boundary]
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        if not isinstance(boundary, Sequence):
            raise EngineError(KrakenEngine.name, f"Unsupported Kraken boundary: {boundary!r}")
        points: list[tuple[float, float]] = []
        for point in boundary:
            if not isinstance(point, Sequence) or isinstance(point, str | bytes) or len(point) != 2:
                raise EngineError(KrakenEngine.name, f"Unsupported Kraken point: {point!r}")
            x, y = point
            points.append((float(x), float(y)))
        return points
