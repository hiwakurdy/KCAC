from __future__ import annotations

from pathlib import Path

from pipeline.config import EngineConfig, PipelineConfig
from pipeline.consensus import build_consensus_page, edit_distance
from pipeline.geometry import rectangle_polygon
from pipeline.models import EnginePageOutput, LineOutput, PageRef


def _config() -> PipelineConfig:
    return PipelineConfig(
        config_path=Path("config.yaml.example"),
        books_jsonl=Path("input/books.jsonl"),
        images_root=Path("ds_test/409/images"),
        output_root=Path("output"),
        engines={name: EngineConfig(enabled=True) for name in ["tesseract", "kraken", "calamari", "qwen2vl", "claude"]},
    )


def test_edit_distance() -> None:
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "axc") == 1
    assert edit_distance("", "abc") == 3


def test_consensus_auto_accepts_three_matching_engines() -> None:
    page = PageRef("kcac_000409", "kcac_000409_p0001", Path("page_0001.jpg"), 100, 100)
    polygon = rectangle_polygon(0, 0, 50, 10)
    outputs = []
    for engine, text in {
        "tesseract": "\u0643\u064a",
        "kraken": "\u06a9\u06cc",
        "calamari": "\u06a9\u06cc",
        "qwen2vl": "\u06a9\u06cc",
        "claude": "\u06a9\u06cc",
    }.items():
        outputs.append(
            EnginePageOutput(
                page_id=page.page_id,
                book_id=page.book_id,
                engine=engine,
                engine_version="test",
                image_filename=page.image_path.name,
                lines=[
                    LineOutput(
                        line_id=f"{page.page_id}_l0001",
                        polygon=polygon,
                        baseline=[(0, 8), (50, 8)],
                        text=text,
                        confidence=0.9,
                        reading_order=1,
                    )
                ],
            )
        )
    consensus = build_consensus_page(page, outputs, _config())
    assert consensus.lines[0].text_normalised == "\u06a9\u06cc"
    assert consensus.lines[0].confidence_label == "auto_accept"
    assert consensus.lines[0].engine_consensus_count == 5
