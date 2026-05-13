from __future__ import annotations

from pathlib import Path

from pipeline.line_sources import lines_from_kcac_json, load_kcac_json_lines
from pipeline.models import PageRef


def test_load_kcac_json_lines_from_sibling_annotationa() -> None:
    page = PageRef(
        book_id="kcac_000409",
        page_id="kcac_000409_p0001",
        image_path=Path("ds_test/409/images/page_0001.jpg"),
        width=1733,
        height=2480,
    )
    lines = load_kcac_json_lines(page)
    assert lines
    assert lines[0].line_id.endswith("_l0001")
    assert lines[0].polygon
    assert lines[0].baseline


def test_text_lines_source_ignores_annotations_lines() -> None:
    data = {
        "annotations": {
            "lines": [
                {
                    "id": "from_annotations",
                    "bbox": [0, 0, 10, 10],
                }
            ]
        },
        "text_lines": [
            {
                "bbox": [20, 30, 40, 10],
                "confidence": 0.7,
            }
        ],
    }
    lines = lines_from_kcac_json(data, "kcac_000409_p0006", source="text_lines")
    assert len(lines) == 1
    assert lines[0].line_id == "kcac_000409_p0006_l0001"
    assert lines[0].polygon[0] == (20.0, 30.0)
    assert lines[0].confidence == 0.7


def test_page_0006_text_lines_count() -> None:
    page = PageRef(
        book_id="kcac_000409",
        page_id="kcac_000409_p0006",
        image_path=Path("ds_test/409/images/page_0006.jpg"),
        width=1803,
        height=2525,
    )
    lines = load_kcac_json_lines(page, source="text_lines")
    assert len(lines) == 21
    assert lines[0].polygon[0] == (194.0, 214.0)
