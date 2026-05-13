from __future__ import annotations

from pipeline.engines.kraken import KrakenEngine


class _KrakenLine:
    def __init__(self) -> None:
        self.boundary = [(1, 2), (10, 2), (10, 8), (1, 8)]


class _KrakenSegmentation:
    def __init__(self) -> None:
        self.lines = [_KrakenLine()]


def test_kraken_segmentation_object_lines_are_supported() -> None:
    lines = KrakenEngine._segmentation_lines(_KrakenSegmentation())

    assert len(lines) == 1
    assert KrakenEngine._boundary_to_polygon(KrakenEngine._line_boundary(lines[0])) == [
        (1.0, 2.0),
        (10.0, 2.0),
        (10.0, 8.0),
        (1.0, 8.0),
    ]
