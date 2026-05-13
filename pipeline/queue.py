from __future__ import annotations

import csv
from pathlib import Path

from .config import PipelineConfig
from .discovery import consensus_output_path, discover_pages
from .jsonio import read_json
from .models import ConsensusPage


def priority_score(page: ConsensusPage) -> float:
    total = len(page.lines)
    disagreement = sum(1 for line in page.lines if line.confidence_label == "disagreement")
    near = sum(1 for line in page.lines if line.confidence_label == "near_agreement")
    return round(disagreement * 10.0 + near * 4.0 + total * 0.25, 3)


def build_annotation_queue(config: PipelineConfig, *, limit: int | None = None) -> Path:
    rows: list[dict[str, object]] = []
    for page_ref in discover_pages(config.images_root, limit=limit):
        path = consensus_output_path(config.output_root, page_ref.book_id, page_ref.page_id)
        if not path.exists():
            continue
        page = ConsensusPage.from_json(read_json(path))
        total_lines = len(page.lines)
        disagreement_lines = sum(1 for line in page.lines if line.confidence_label == "disagreement")
        rows.append(
            {
                "page_id": page.page_id,
                "book_id": page.book_id,
                "priority_score": priority_score(page),
                "disagreement_lines": disagreement_lines,
                "total_lines": total_lines,
                "era": "unknown",
                "script": "sorani-arabic",
                "typography": "unknown",
                "assigned_to": "",
                "status": "todo",
            }
        )
    def sort_key(row: dict[str, object]) -> tuple[float, str, str]:
        return -float(str(row["priority_score"])), str(row["book_id"]), str(row["page_id"])

    rows.sort(key=sort_key)
    out_path = config.output_root / "annotation_queue.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "page_id",
        "book_id",
        "priority_score",
        "disagreement_lines",
        "total_lines",
        "era",
        "script",
        "typography",
        "assigned_to",
        "status",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path
