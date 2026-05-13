from __future__ import annotations

import json
from pathlib import Path


def test_existing_kcac_json_sample_shape() -> None:
    path = Path("ds_test/409/annotationa/page_0001.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["file"] == "page_0001.jpg"
    assert data["bbox_format"] == "coco_xywh"
    assert data["width"] > 0
    assert data["height"] > 0
    assert data["text_lines"]
    first_line = data["text_lines"][0]
    assert len(first_line["bbox"]) == 4
    assert 0 <= first_line["confidence"] <= 1
