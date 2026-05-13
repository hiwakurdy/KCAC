from __future__ import annotations

from pathlib import Path
from typing import Any

from .geometry import baseline_from_polygon, rectangle_polygon
from .jsonio import read_json
from .models import LineOutput, PageRef, Polygon


def candidate_annotation_paths(page: PageRef, annotations_root: Path | None = None) -> list[Path]:
    names = [f"{page.image_path.stem}.json", f"{page.page_id}.json"]
    roots: list[Path] = []
    if annotations_root is not None:
        roots.append(annotations_root)
    parent = page.image_path.parent.parent
    roots.extend([parent / "annotations", parent / "annotationa", parent / "json", parent / "labels"])
    candidates: list[Path] = []
    for root in roots:
        for name in names:
            candidates.append(root / name)
    return candidates


def load_kcac_json_lines(
    page: PageRef,
    annotations_root: Path | None = None,
    *,
    source: str = "auto",
) -> list[LineOutput]:
    for path in candidate_annotation_paths(page, annotations_root):
        if path.exists():
            return lines_from_kcac_json(read_json(path), page.page_id, source=source)
    return []


def lines_from_kcac_json(data: dict[str, Any], page_id: str, *, source: str = "auto") -> list[LineOutput]:
    items = _line_items(data, source)
    lines: list[LineOutput] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        polygon = _polygon_from_item(item)
        if not polygon:
            continue
        baseline = _points(item.get("baseline")) or baseline_from_polygon(polygon)
        transcription = item.get("transcription", {})
        text = ""
        if isinstance(transcription, dict):
            text = str(transcription.get("raw") or transcription.get("normalised") or "")
        text = str(item.get("text") or item.get("text_raw") or text)
        confidence = item.get("confidence", item.get("score"))
        lines.append(
            LineOutput(
                line_id=str(item.get("line_id") or item.get("id") or f"{page_id}_l{index:04d}"),
                polygon=polygon,
                baseline=baseline,
                text=text,
                confidence=None if confidence is None else float(confidence),
                reading_order=int(item.get("reading_order") or index),
            )
        )
    lines.sort(key=lambda line: line.reading_order or 0)
    return lines


def _line_items(data: dict[str, Any], source: str) -> list[Any]:
    source = source.lower()
    if source == "text_lines":
        items = data.get("text_lines", [])
    elif source in {"annotations", "annotations.lines"}:
        items = data.get("annotations", {}).get("lines", [])
    elif source in {"lines", "legacy_lines"}:
        items = data.get("lines", [])
    elif source in {"auto", "kcac_json", "json"}:
        items = data.get("annotations", {}).get("lines") or data.get("lines") or data.get("text_lines") or []
    else:
        raise ValueError(f"Unsupported KCAC JSON line source: {source}")
    return list(items) if isinstance(items, list) else []


def _polygon_from_item(item: dict[str, Any]) -> Polygon:
    polygon = _points(item.get("polygon"))
    if polygon:
        return polygon
    bbox = item.get("bbox")
    if isinstance(bbox, list | tuple) and len(bbox) == 4:
        x, y, width, height = [float(value) for value in bbox]
        return rectangle_polygon(x, y, width, height)
    return []


def _points(value: object) -> Polygon:
    if not isinstance(value, list | tuple):
        return []
    points: Polygon = []
    for point in value:
        if not isinstance(point, list | tuple) or len(point) != 2:
            return []
        x, y = point
        points.append((float(x), float(y)))
    return points
