from __future__ import annotations

import json
from pathlib import Path

from pipeline import bootstrap as bootstrap_module
from pipeline.config import EngineConfig, PipelineConfig
from pipeline.models import EngineFailure, EnginePageOutput, PageRef


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        config_path=tmp_path / "config.yaml",
        books_jsonl=tmp_path / "books.jsonl",
        images_root=tmp_path / "images",
        output_root=tmp_path / "output",
        engines={"bad": EngineConfig(enabled=True)},
    )


def test_bootstrap_skips_engine_after_nonrecoverable_failure(monkeypatch, tmp_path: Path) -> None:
    pages = [
        PageRef("book", "book_p0001", tmp_path / "p1.png"),
        PageRef("book", "book_p0002", tmp_path / "p2.png"),
    ]
    calls = {"count": 0}

    def fake_run_engine(page: PageRef, engine_name: str, config: PipelineConfig) -> EnginePageOutput:
        calls["count"] += 1
        return EnginePageOutput(
            page_id=page.page_id,
            book_id=page.book_id,
            engine=engine_name,
            engine_version="",
            image_filename=page.image_path.name,
            failures=[
                EngineFailure(
                    engine=engine_name,
                    page_id=page.page_id,
                    error_type="EngineError",
                    message="missing setup",
                    recoverable=False,
                )
            ],
        )

    monkeypatch.setattr(bootstrap_module, "discover_pages", lambda _root, limit=None: pages[:limit])
    monkeypatch.setattr(bootstrap_module, "run_engine", fake_run_engine)

    paths = bootstrap_module.bootstrap_pages(_config(tmp_path), limit=2)

    assert calls["count"] == 1
    assert len(paths) == 2
    skipped = json.loads(paths[1].read_text(encoding="utf-8"))
    assert skipped["failures"][0]["error_type"] == "EngineSkipped"
    assert skipped["failures"][0]["recoverable"] is False
