from __future__ import annotations

from pipeline.geometry import polygon_iou, rectangle_polygon


def test_polygon_iou_for_overlapping_rectangles() -> None:
    first = rectangle_polygon(0, 0, 10, 10)
    second = rectangle_polygon(5, 0, 10, 10)
    assert round(polygon_iou(first, second), 4) == 0.3333


def test_polygon_iou_for_disjoint_rectangles() -> None:
    first = rectangle_polygon(0, 0, 10, 10)
    second = rectangle_polygon(20, 20, 5, 5)
    assert polygon_iou(first, second) == 0.0
