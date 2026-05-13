from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from .models import PageRef

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def normalise_book_id(raw: str) -> str:
    if raw.startswith("kcac_"):
        return raw
    if raw.isdigit():
        return f"kcac_{int(raw):06d}"
    return re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_") or "kcac_000000"


def page_sequence_from_name(stem: str, fallback: int) -> int:
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        return fallback
    return int(numbers[-1])


def discover_pages(images_root: Path, *, limit: int | None = None) -> list[PageRef]:
    if not images_root.exists():
        raise FileNotFoundError(f"Images root does not exist: {images_root}")

    files = sorted(path for path in images_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    pages: list[PageRef] = []
    for idx, image_path in enumerate(files, start=1):
        parent = image_path.parent.name
        book_id = normalise_book_id(parent if image_path.parent != images_root else images_root.name)
        sequence = page_sequence_from_name(image_path.stem, idx)
        page_id = f"{book_id}_p{sequence:04d}"
        width: int | None = None
        height: int | None = None
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError:
            pass
        pages.append(PageRef(book_id=book_id, page_id=page_id, image_path=image_path, width=width, height=height))
        if limit is not None and len(pages) >= limit:
            break
    return pages


def raw_engine_output_path(output_root: Path, book_id: str, page_id: str, engine: str) -> Path:
    return output_root / "ocr_raw" / book_id / page_id / f"{engine}.json"


def consensus_output_path(output_root: Path, book_id: str, page_id: str) -> Path:
    return output_root / "consensus" / book_id / f"{page_id}.json"


def pagexml_output_path(output_root: Path, book_id: str, page_id: str) -> Path:
    return output_root / "page_xml" / book_id / f"{page_id}.xml"
