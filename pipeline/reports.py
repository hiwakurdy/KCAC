from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path

from .config import PipelineConfig
from .consensus import edit_distance
from .discovery import consensus_output_path, discover_pages
from .jsonio import read_json
from .models import ConfidenceLabel, ConsensusPage


def _load_pages(config: PipelineConfig, limit: int | None = None) -> list[ConsensusPage]:
    pages: list[ConsensusPage] = []
    for page_ref in discover_pages(config.images_root, limit=limit):
        path = consensus_output_path(config.output_root, page_ref.book_id, page_ref.page_id)
        if path.exists():
            pages.append(ConsensusPage.from_json(read_json(path)))
    return pages


def write_confidence_histogram(pages: list[ConsensusPage], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(line.confidence_label for page in pages for line in page.lines)
    try:
        import matplotlib.pyplot as plt

        labels: list[ConfidenceLabel] = ["auto_accept", "near_agreement", "disagreement"]
        values = [counts.get(label, 0) for label in labels]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, values, color=["#3fb950", "#d29922", "#f85149"])
        ax.set_ylabel("Lines")
        ax.set_title("Consensus Confidence")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except ImportError:
        path.with_suffix(".txt").write_text("\n".join(f"{k},{v}" for k, v in counts.items()), encoding="utf-8")


def write_disagreement_matrix(pages: list[ConsensusPage], path: Path) -> None:
    totals: dict[tuple[str, str], list[int]] = defaultdict(list)
    for page in pages:
        for line in page.lines:
            for first, second in combinations(sorted(line.engine_outputs), 2):
                totals[(first, second)].append(edit_distance(line.engine_outputs[first], line.engine_outputs[second]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["engine_a", "engine_b", "line_pairs", "mean_edit_distance", "exact_agreement_rate"])
        for (first, second), distances in sorted(totals.items()):
            exact = sum(1 for distance in distances if distance == 0)
            mean = sum(distances) / len(distances) if distances else 0.0
            writer.writerow([first, second, len(distances), round(mean, 4), round(exact / len(distances), 4) if distances else 0.0])


def write_coverage_matrix(pages: list[ConsensusPage], path: Path) -> None:
    by_book = Counter(page.book_id for page in pages)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["book_id", "pages", "era", "script", "typography"])
        for book_id, pages_count in sorted(by_book.items()):
            writer.writerow([book_id, pages_count, "unknown", "sorani-arabic", "unknown"])


def write_daily_summary(pages: list[ConsensusPage], path: Path) -> None:
    total_lines = sum(len(page.lines) for page in pages)
    labels = Counter(line.confidence_label for page in pages for line in page.lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"# KCAC Daily OCR Summary {date.today().isoformat()}",
                "",
                f"- Pages with consensus: {len(pages)}",
                f"- Lines: {total_lines}",
                f"- Auto accept: {labels.get('auto_accept', 0)}",
                f"- Near agreement: {labels.get('near_agreement', 0)}",
                f"- Disagreement: {labels.get('disagreement', 0)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_reports(config: PipelineConfig, *, limit: int | None = None) -> list[Path]:
    pages = _load_pages(config, limit=limit)
    reports_root = config.output_root / "reports"
    histogram = reports_root / "line_confidence_histogram.png"
    disagreement = reports_root / "disagreement_matrix.csv"
    coverage = reports_root / "coverage_matrix.csv"
    daily = reports_root / f"daily_{date.today().isoformat()}.md"
    write_confidence_histogram(pages, histogram)
    write_disagreement_matrix(pages, disagreement)
    write_coverage_matrix(pages, coverage)
    if config.daily_report:
        write_daily_summary(pages, daily)
    return [histogram, disagreement, coverage, daily]
