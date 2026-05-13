from __future__ import annotations

import shutil
from pathlib import Path

from .config import PipelineConfig
from .discovery import discover_pages, pagexml_output_path
from .jsonio import write_json


def build_escriptorium_import(config: PipelineConfig, *, limit: int | None = None) -> Path:
    root = config.output_root / "escriptorium_import"
    pages = discover_pages(config.images_root, limit=limit)
    manifest_pages: list[dict[str, str]] = []
    for page in pages:
        xml_path = pagexml_output_path(config.output_root, page.book_id, page.page_id)
        if not xml_path.exists():
            continue
        target_dir = root / page.book_id
        target_dir.mkdir(parents=True, exist_ok=True)
        image_target = target_dir / page.image_path.name
        xml_target = target_dir / xml_path.name
        shutil.copy2(page.image_path, image_target)
        shutil.copy2(xml_path, xml_target)
        manifest_pages.append(
            {
                "book_id": page.book_id,
                "page_id": page.page_id,
                "image": str(image_target.relative_to(root)),
                "page_xml": str(xml_target.relative_to(root)),
            }
        )
    write_json(
        root / "manifest.json",
        {
            "name": "KCAC OCR Pipeline eScriptorium Import",
            "format": "image-pagexml-sidecar",
            "pages": manifest_pages,
        },
    )
    return root
