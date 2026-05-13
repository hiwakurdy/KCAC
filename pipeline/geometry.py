from __future__ import annotations

from collections.abc import Sequence

from .models import Point, Polygon


def polygon_area(poly: Sequence[Point]) -> float:
    if len(poly) < 3:
        return 0.0
    total = 0.0
    for idx, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(idx + 1) % len(poly)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _signed_area(poly: Sequence[Point]) -> float:
    total = 0.0
    for idx, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(idx + 1) % len(poly)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _inside(point: Point, edge_start: Point, edge_end: Point, ccw: bool) -> bool:
    px, py = point
    ax, ay = edge_start
    bx, by = edge_end
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    return cross >= -1e-9 if ccw else cross <= 1e-9


def _intersection(p1: Point, p2: Point, e1: Point, e2: Point) -> Point:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = e1
    x4, y4 = e2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return px, py


def convex_intersection(subject: Polygon, clip: Polygon) -> Polygon:
    if len(subject) < 3 or len(clip) < 3:
        return []
    output: Polygon = list(subject)
    ccw = _signed_area(clip) >= 0
    for idx, edge_start in enumerate(clip):
        edge_end = clip[(idx + 1) % len(clip)]
        input_poly = output
        output = []
        if not input_poly:
            break
        previous = input_poly[-1]
        for current in input_poly:
            current_inside = _inside(current, edge_start, edge_end, ccw)
            previous_inside = _inside(previous, edge_start, edge_end, ccw)
            if current_inside:
                if not previous_inside:
                    output.append(_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(_intersection(previous, current, edge_start, edge_end))
            previous = current
    return output


def polygon_iou(first: Polygon, second: Polygon) -> float:
    first_area = polygon_area(first)
    second_area = polygon_area(second)
    if first_area <= 0 or second_area <= 0:
        return 0.0
    intersection = polygon_area(convex_intersection(first, second))
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def bbox_from_polygon(poly: Sequence[Point]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in poly]
    ys = [point[1] for point in poly]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return x1, y1, x2 - x1, y2 - y1


def points_string(poly: Sequence[Point]) -> str:
    return " ".join(f"{round(x, 2)},{round(y, 2)}" for x, y in poly)


def rectangle_polygon(x: float, y: float, width: float, height: float) -> Polygon:
    return [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]


def baseline_from_polygon(poly: Sequence[Point]) -> list[Point]:
    x, y, width, height = bbox_from_polygon(poly)
    baseline_y = y + height * 0.78
    return [(x, baseline_y), (x + width, baseline_y)]
