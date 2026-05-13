from __future__ import annotations

from collections import Counter
from pathlib import Path

from .config import PipelineConfig
from .discovery import consensus_output_path, discover_pages, raw_engine_output_path
from .geometry import polygon_iou
from .jsonio import read_json, write_json
from .models import ConsensusLine, ConsensusPage, EnginePageOutput, LineOutput, NormalisationChange, PageRef
from .normalise import normalise_sorani


def edit_distance(first: str, second: str) -> int:
    if first == second:
        return 0
    if not first:
        return len(second)
    if not second:
        return len(first)
    previous = list(range(len(second) + 1))
    for i, char_first in enumerate(first, start=1):
        current = [i]
        for j, char_second in enumerate(second, start=1):
            cost = 0 if char_first == char_second else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def max_pairwise_distance(texts: list[str]) -> int:
    if len(texts) < 2:
        return 0
    max_distance = 0
    for i, first in enumerate(texts):
        for second in texts[i + 1 :]:
            max_distance = max(max_distance, edit_distance(first, second))
    return max_distance


def choose_consensus_text(texts_by_engine: dict[str, str], preferred_order: list[str]) -> tuple[str, int]:
    non_empty = {engine: text for engine, text in texts_by_engine.items() if text}
    if not non_empty:
        return "", 0
    counts = Counter(non_empty.values())
    highest = max(counts.values())
    candidates = {text for text, count in counts.items() if count == highest}
    for engine in preferred_order:
        text = non_empty.get(engine)
        if text in candidates:
            return text, highest
    return sorted(candidates)[0], highest


def confidence_label(
    texts: list[str],
    consensus_count: int,
    *,
    min_agreeing: int,
    auto_accept_max_edits: int,
    near_agreement_max_edits: int,
) -> str:
    max_distance = max_pairwise_distance(texts)
    if consensus_count >= min_agreeing and max_distance <= auto_accept_max_edits:
        return "auto_accept"
    close_count = sum(1 for text in texts if texts and edit_distance(text, texts[0]) <= near_agreement_max_edits)
    if consensus_count >= min_agreeing or close_count >= min_agreeing:
        return "near_agreement"
    return "disagreement"


class _Cluster:
    def __init__(self, first_engine: str, first_line: LineOutput) -> None:
        self.lines: dict[str, LineOutput] = {first_engine: first_line}

    @property
    def representative(self) -> LineOutput:
        return next(iter(self.lines.values()))

    def best_iou(self, line: LineOutput) -> float:
        return max(polygon_iou(existing.polygon, line.polygon) for existing in self.lines.values())


def cluster_engine_lines(outputs: list[EnginePageOutput], iou_threshold: float) -> list[_Cluster]:
    clusters: list[_Cluster] = []
    for output in outputs:
        for line in output.lines:
            best_cluster: _Cluster | None = None
            best_score = 0.0
            for cluster in clusters:
                score = cluster.best_iou(line)
                if score > best_score:
                    best_score = score
                    best_cluster = cluster
            if best_cluster is not None and best_score >= iou_threshold:
                best_cluster.lines[output.engine] = line
            else:
                clusters.append(_Cluster(output.engine, line))
    return clusters


def build_consensus_page(page: PageRef, outputs: list[EnginePageOutput], config: PipelineConfig) -> ConsensusPage:
    preferred_order = list(config.engines.keys())
    failed_engines = {output.engine: [failure.to_json() for failure in output.failures] for output in outputs if output.failures}
    lines: list[ConsensusLine] = []
    for order, cluster in enumerate(cluster_engine_lines(outputs, config.iou_threshold), start=1):
        engine_outputs = {engine: line.text for engine, line in cluster.lines.items()}
        engine_confidences = {engine: line.confidence for engine, line in cluster.lines.items()}
        normalised_by_engine = {
            engine: normalise_sorani(text, strip_tashkeel=config.strip_tashkeel, unicode_form=config.unicode_form).normalised
            for engine, text in engine_outputs.items()
        }
        consensus_text, consensus_count = choose_consensus_text(normalised_by_engine, preferred_order)
        normalised = normalise_sorani(consensus_text, strip_tashkeel=config.strip_tashkeel, unicode_form=config.unicode_form)
        texts = list(normalised_by_engine.values())
        label = confidence_label(
            texts,
            consensus_count,
            min_agreeing=config.auto_accept_min_agreeing_engines,
            auto_accept_max_edits=config.auto_accept_max_pairwise_edits,
            near_agreement_max_edits=config.near_agreement_max_edits,
        )
        rep = cluster.representative
        trace: list[NormalisationChange] = normalised.changes
        lines.append(
            ConsensusLine(
                line_id=rep.line_id or f"{page.page_id}_l{order:04d}",
                polygon=rep.polygon,
                baseline=rep.baseline,
                text_raw=choose_consensus_text(engine_outputs, preferred_order)[0],
                text_normalised=normalised.normalised,
                normalisation_trace=trace,
                confidence_label=label,  # type: ignore[arg-type]
                engine_outputs=engine_outputs,
                engine_confidences=engine_confidences,
                engine_consensus_count=consensus_count,
                max_pairwise_distance=max_pairwise_distance(texts),
                reading_order=rep.reading_order or order,
            )
        )
    lines.sort(key=lambda line: line.reading_order)
    return ConsensusPage(
        page_id=page.page_id,
        book_id=page.book_id,
        image_filename=page.image_path.name,
        width=page.width,
        height=page.height,
        lines=lines,
        failed_engines=failed_engines,
    )


def consensus_for_pages(config: PipelineConfig, *, limit: int | None = None, force: bool = False) -> list[Path]:
    written: list[Path] = []
    pages = discover_pages(config.images_root, limit=limit)
    for page in pages:
        out_path = consensus_output_path(config.output_root, page.book_id, page.page_id)
        if out_path.exists() and not force:
            written.append(out_path)
            continue
        outputs: list[EnginePageOutput] = []
        for engine_name in config.engines:
            raw_path = raw_engine_output_path(config.output_root, page.book_id, page.page_id, engine_name)
            if raw_path.exists():
                outputs.append(EnginePageOutput.from_json(read_json(raw_path)))
        consensus = build_consensus_page(page, outputs, config)
        write_json(out_path, consensus.to_json())
        written.append(out_path)
    return written
