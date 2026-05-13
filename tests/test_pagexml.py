from __future__ import annotations

import xml.etree.ElementTree as ET

from pipeline.geometry import rectangle_polygon
from pipeline.models import ConsensusLine, ConsensusPage
from pipeline.pagexml_export import PAGE_NS, page_to_xml


def test_pagexml_contains_raw_and_normalised_text_equiv() -> None:
    page = ConsensusPage(
        page_id="kcac_000409_p0001",
        book_id="kcac_000409",
        image_filename="page_0001.jpg",
        width=100,
        height=200,
        lines=[
            ConsensusLine(
                line_id="kcac_000409_p0001_l0001",
                polygon=rectangle_polygon(0, 0, 50, 10),
                baseline=[(0, 8), (50, 8)],
                text_raw="\u0643\u064a",
                text_normalised="\u06a9\u06cc",
                normalisation_trace=[],
                confidence_label="auto_accept",
                engine_outputs={"tesseract": "\u0643\u064a"},
                engine_confidences={"tesseract": 0.9},
                engine_consensus_count=1,
                max_pairwise_distance=0,
                reading_order=1,
            )
        ],
    )
    root = page_to_xml(page).getroot()
    ET.tostring(root)
    text_equivs = root.findall(f".//{{{PAGE_NS}}}TextEquiv")
    assert [item.attrib["index"] for item in text_equivs] == ["1", "2"]
    assert text_equivs[0].find(f"{{{PAGE_NS}}}Unicode").text == "\u0643\u064a"
    assert text_equivs[1].find(f"{{{PAGE_NS}}}Unicode").text == "\u06a9\u06cc"
