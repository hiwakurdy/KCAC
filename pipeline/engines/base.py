from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

from PIL import Image

from ..config import EngineConfig
from ..geometry import bbox_from_polygon, rectangle_polygon
from ..models import EnginePageOutput, LineOutput, PageRef, Polygon


class EngineError(RuntimeError):
    def __init__(self, engine: str, message: str, *, recoverable: bool = True) -> None:
        super().__init__(message)
        self.engine = engine
        self.recoverable = recoverable


class BaseOcrEngine(ABC):
    name: str

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    @abstractmethod
    def version(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def run(self, page: PageRef) -> EnginePageOutput:
        raise NotImplementedError

    def require_import(self, module_name: str) -> Any:
        try:
            return __import__(module_name)
        except ImportError as exc:
            raise EngineError(
                self.name,
                f"Python package '{module_name}' is required for the {self.name} engine",
                recoverable=False,
            ) from exc


def line_id(page_id: str, index: int) -> str:
    return f"{page_id}_l{index:04d}"


def polygon_from_bbox(x: float, y: float, width: float, height: float) -> Polygon:
    return rectangle_polygon(float(x), float(y), float(width), float(height))


def crop_line(image_path: Path, polygon: Polygon) -> Image.Image:
    with Image.open(image_path) as image:
        x, y, width, height = bbox_from_polygon(polygon)
        left = max(0, int(x))
        top = max(0, int(y))
        right = min(image.width, int(x + width))
        bottom = min(image.height, int(y + height))
        return cast(Image.Image, image.crop((left, top, right, bottom)).convert("RGB"))


def image_to_base64_jpeg(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_output(page: PageRef, engine: str, version: str, lines: list[LineOutput]) -> EnginePageOutput:
    return EnginePageOutput(
        page_id=page.page_id,
        book_id=page.book_id,
        engine=engine,
        engine_version=version,
        image_filename=page.image_path.name,
        lines=lines,
        metadata={"image_path": str(page.image_path), "width": page.width, "height": page.height},
    )
