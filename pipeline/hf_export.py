from __future__ import annotations

import json
from pathlib import Path

from .config import PipelineConfig
from .discovery import consensus_output_path, discover_pages
from .jsonio import read_json
from .models import ConsensusPage


def export_hf_jsonl(config: PipelineConfig, *, limit: int | None = None) -> Path:
    out_path = config.output_root / "hf_dataset" / "lines.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for page_ref in discover_pages(config.images_root, limit=limit):
            consensus_path = consensus_output_path(config.output_root, page_ref.book_id, page_ref.page_id)
            if not consensus_path.exists():
                continue
            page = ConsensusPage.from_json(read_json(consensus_path))
            for line in page.lines:
                row = {
                    "id": line.line_id,
                    "book_id": page.book_id,
                    "page_id": page.page_id,
                    "image": page.image_filename,
                    "polygon": line.polygon,
                    "baseline": line.baseline,
                    "text_raw": line.text_raw,
                    "text_normalised": line.text_normalised,
                    "confidence_label": line.confidence_label,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path
