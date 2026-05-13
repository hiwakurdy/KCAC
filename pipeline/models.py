from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

JsonDict = dict[str, Any]
Point = tuple[float, float]
Polygon = list[Point]
Baseline = list[Point]
ConfidenceLabel = Literal["auto_accept", "near_agreement", "disagreement"]


@dataclass(slots=True)
class PageRef:
    book_id: str
    page_id: str
    image_path: Path
    width: int | None = None
    height: int | None = None

    def to_json(self) -> JsonDict:
        data = asdict(self)
        data["image_path"] = str(self.image_path)
        return data


@dataclass(slots=True)
class LineOutput:
    line_id: str
    polygon: Polygon
    baseline: Baseline
    text: str
    confidence: float | None = None
    text_normalised: str | None = None
    reading_order: int | None = None

    @classmethod
    def from_json(cls, data: JsonDict) -> LineOutput:
        return cls(
            line_id=str(data.get("line_id") or data.get("id") or ""),
            polygon=[(float(x), float(y)) for x, y in data.get("polygon", [])],
            baseline=[(float(x), float(y)) for x, y in data.get("baseline", [])],
            text=str(data.get("text") or data.get("text_raw") or ""),
            confidence=None if data.get("confidence") is None else float(data["confidence"]),
            text_normalised=data.get("text_normalised"),
            reading_order=None if data.get("reading_order") is None else int(data["reading_order"]),
        )

    def to_json(self) -> JsonDict:
        return {
            "line_id": self.line_id,
            "polygon": [[x, y] for x, y in self.polygon],
            "baseline": [[x, y] for x, y in self.baseline],
            "text": self.text,
            "text_normalised": self.text_normalised,
            "confidence": self.confidence,
            "reading_order": self.reading_order,
        }


@dataclass(slots=True)
class EngineFailure:
    engine: str
    page_id: str
    error_type: str
    message: str
    recoverable: bool = True

    def to_json(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class EnginePageOutput:
    page_id: str
    book_id: str
    engine: str
    engine_version: str
    image_filename: str
    lines: list[LineOutput] = field(default_factory=list)
    failures: list[EngineFailure] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: JsonDict) -> EnginePageOutput:
        return cls(
            page_id=str(data["page_id"]),
            book_id=str(data.get("book_id") or data["page_id"].split("_p")[0]),
            engine=str(data["engine"]),
            engine_version=str(data.get("engine_version", "")),
            image_filename=str(data.get("image_filename", "")),
            lines=[LineOutput.from_json(item) for item in data.get("lines", [])],
            failures=[
                EngineFailure(
                    engine=str(item.get("engine", data.get("engine", ""))),
                    page_id=str(item.get("page_id", data.get("page_id", ""))),
                    error_type=str(item.get("error_type", "EngineError")),
                    message=str(item.get("message", "")),
                    recoverable=bool(item.get("recoverable", True)),
                )
                for item in data.get("failures", [])
            ],
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self) -> JsonDict:
        return {
            "page_id": self.page_id,
            "book_id": self.book_id,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "image_filename": self.image_filename,
            "lines": [line.to_json() for line in self.lines],
            "failures": [failure.to_json() for failure in self.failures],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class NormalisationChange:
    index: int
    before: str
    after: str
    reason: str

    def to_json(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class NormalisedText:
    raw: str
    normalised: str
    changes: list[NormalisationChange]

    def to_json(self) -> JsonDict:
        return {
            "raw": self.raw,
            "normalised": self.normalised,
            "changes": [change.to_json() for change in self.changes],
        }


@dataclass(slots=True)
class ConsensusLine:
    line_id: str
    polygon: Polygon
    baseline: Baseline
    text_raw: str
    text_normalised: str
    normalisation_trace: list[NormalisationChange]
    confidence_label: ConfidenceLabel
    engine_outputs: dict[str, str]
    engine_confidences: dict[str, float | None]
    engine_consensus_count: int
    max_pairwise_distance: int
    reading_order: int

    def to_json(self) -> JsonDict:
        return {
            "line_id": self.line_id,
            "polygon": [[x, y] for x, y in self.polygon],
            "baseline": [[x, y] for x, y in self.baseline],
            "text_raw": self.text_raw,
            "text_normalised": self.text_normalised,
            "normalisation_trace": [change.to_json() for change in self.normalisation_trace],
            "confidence_label": self.confidence_label,
            "engine_outputs": self.engine_outputs,
            "engine_confidences": self.engine_confidences,
            "engine_consensus_count": self.engine_consensus_count,
            "max_pairwise_distance": self.max_pairwise_distance,
            "reading_order": self.reading_order,
        }


@dataclass(slots=True)
class ConsensusPage:
    page_id: str
    book_id: str
    image_filename: str
    width: int | None
    height: int | None
    lines: list[ConsensusLine]
    failed_engines: dict[str, list[JsonDict]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: JsonDict) -> ConsensusPage:
        lines: list[ConsensusLine] = []
        for item in data.get("lines", []):
            trace = [
                NormalisationChange(
                    index=int(change["index"]),
                    before=str(change["before"]),
                    after=str(change["after"]),
                    reason=str(change["reason"]),
                )
                for change in item.get("normalisation_trace", [])
            ]
            lines.append(
                ConsensusLine(
                    line_id=str(item["line_id"]),
                    polygon=[(float(x), float(y)) for x, y in item.get("polygon", [])],
                    baseline=[(float(x), float(y)) for x, y in item.get("baseline", [])],
                    text_raw=str(item.get("text_raw", item.get("text_consensus", ""))),
                    text_normalised=str(item.get("text_normalised", item.get("text_consensus", ""))),
                    normalisation_trace=trace,
                    confidence_label=item.get("confidence_label", "disagreement"),
                    engine_outputs={str(k): str(v) for k, v in item.get("engine_outputs", {}).items()},
                    engine_confidences=dict(item.get("engine_confidences", {})),
                    engine_consensus_count=int(item.get("engine_consensus_count", 0)),
                    max_pairwise_distance=int(item.get("max_pairwise_distance", 0)),
                    reading_order=int(item.get("reading_order", len(lines) + 1)),
                )
            )
        return cls(
            page_id=str(data["page_id"]),
            book_id=str(data.get("book_id") or data["page_id"].split("_p")[0]),
            image_filename=str(data.get("image_filename", "")),
            width=data.get("width"),
            height=data.get("height"),
            lines=lines,
            failed_engines=dict(data.get("failed_engines", {})),
        )

    def to_json(self) -> JsonDict:
        return {
            "page_id": self.page_id,
            "book_id": self.book_id,
            "image_filename": self.image_filename,
            "width": self.width,
            "height": self.height,
            "lines": [line.to_json() for line in self.lines],
            "failed_engines": self.failed_engines,
        }


@dataclass(slots=True)
class BudgetEvent:
    engine: str
    book_id: str
    page_id: str
    units: int
    estimated_usd: float
    description: str

    def to_json(self) -> JsonDict:
        return asdict(self)
