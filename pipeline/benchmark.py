from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .config import PipelineConfig
from .consensus import edit_distance
from .discovery import consensus_output_path, discover_pages, raw_engine_output_path
from .jsonio import read_json, write_json
from .models import ConsensusPage, EnginePageOutput


def cer(prediction: str, truth: str) -> float:
    if not truth:
        return 0.0 if not prediction else 1.0
    return edit_distance(prediction, truth) / len(truth)


def wer(prediction: str, truth: str) -> float:
    pred_words = prediction.split()
    truth_words = truth.split()
    if not truth_words:
        return 0.0 if not pred_words else 1.0
    return edit_distance("\n".join(pred_words), "\n".join(truth_words)) / len(truth_words)


def benchmark(config: PipelineConfig, *, limit: int | None = None) -> Path:
    scores: dict[str, list[dict[str, float]]] = defaultdict(list)
    for page_ref in discover_pages(config.images_root, limit=limit):
        consensus_path = consensus_output_path(config.output_root, page_ref.book_id, page_ref.page_id)
        if not consensus_path.exists():
            continue
        gold = ConsensusPage.from_json(read_json(consensus_path))
        gold_by_line = {line.line_id: line.text_normalised for line in gold.lines}
        for engine_name in config.engines:
            raw_path = raw_engine_output_path(config.output_root, page_ref.book_id, page_ref.page_id, engine_name)
            if not raw_path.exists():
                continue
            output = EnginePageOutput.from_json(read_json(raw_path))
            for line in output.lines:
                truth = gold_by_line.get(line.line_id)
                if truth is None:
                    continue
                scores[engine_name].append(
                    {
                        "cer": cer(line.text_normalised or line.text, truth),
                        "wer": wer(line.text_normalised or line.text, truth),
                        "line_accuracy": 1.0 if (line.text_normalised or line.text) == truth else 0.0,
                    }
                )

    engine_results: dict[str, dict[str, object]] = {}
    results: dict[str, object] = {"engines": engine_results}
    for engine, values in sorted(scores.items()):
        count = len(values)
        engine_results[engine] = {
            "lines": count,
            "cer": round(sum(item["cer"] for item in values) / count, 6) if count else None,
            "wer": round(sum(item["wer"] for item in values) / count, 6) if count else None,
            "line_accuracy": round(sum(item["line_accuracy"] for item in values) / count, 6) if count else None,
            "buckets": {
                "script:sorani-arabic": count,
                "era:unknown": count,
                "typography:unknown": count,
            },
        }
    out_path = config.output_root / "benchmark" / "baseline_results.json"
    write_json(out_path, results)
    report = config.output_root / "benchmark" / "baseline_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# Baseline OCR Benchmark\n\n```json\n" + json.dumps(results, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    return out_path
