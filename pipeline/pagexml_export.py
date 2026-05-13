from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .config import PipelineConfig
from .discovery import consensus_output_path, discover_pages, pagexml_output_path
from .geometry import points_string
from .jsonio import read_json
from .models import ConsensusPage

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
SCHEMA_LOCATION = f"{PAGE_NS} http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"

ET.register_namespace("", PAGE_NS)
ET.register_namespace("xsi", XSI_NS)


def _text_equiv(parent: ET.Element, index: int, text: str, data_type: str) -> None:
    equiv = ET.SubElement(parent, f"{{{PAGE_NS}}}TextEquiv", {"index": str(index), "dataType": data_type})
    unicode_el = ET.SubElement(equiv, f"{{{PAGE_NS}}}Unicode")
    unicode_el.text = text


def page_to_xml(page: ConsensusPage) -> ET.ElementTree:
    root = ET.Element(
        f"{{{PAGE_NS}}}PcGts",
        {
            f"{{{XSI_NS}}}schemaLocation": SCHEMA_LOCATION,
        },
    )
    metadata = ET.SubElement(root, f"{{{PAGE_NS}}}Metadata")
    ET.SubElement(metadata, f"{{{PAGE_NS}}}Creator").text = "KCAC OCR Pipeline"
    page_el = ET.SubElement(
        root,
        f"{{{PAGE_NS}}}Page",
        {
            "imageFilename": page.image_filename,
            "imageWidth": str(page.width or 0),
            "imageHeight": str(page.height or 0),
            "readingDirection": "right-to-left",
        },
    )
    region = ET.SubElement(page_el, f"{{{PAGE_NS}}}TextRegion", {"id": f"{page.page_id}_r0001"})
    if page.width and page.height:
        ET.SubElement(region, f"{{{PAGE_NS}}}Coords", {"points": f"0,0 {page.width},0 {page.width},{page.height} 0,{page.height}"})
    for line in page.lines:
        custom = f"confidence_label:{line.confidence_label};engine_consensus_count:{line.engine_consensus_count}"
        line_el = ET.SubElement(
            region,
            f"{{{PAGE_NS}}}TextLine",
            {"id": line.line_id, "custom": custom},
        )
        ET.SubElement(line_el, f"{{{PAGE_NS}}}Coords", {"points": points_string(line.polygon)})
        if line.baseline:
            ET.SubElement(line_el, f"{{{PAGE_NS}}}Baseline", {"points": points_string(line.baseline)})
        _text_equiv(line_el, 1, line.text_raw, "raw")
        _text_equiv(line_el, 2, line.text_normalised, "normalised")
    return ET.ElementTree(root)


def write_page_xml(path: Path, page: ConsensusPage) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = page_to_xml(page)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def export_pagexml(config: PipelineConfig, *, limit: int | None = None, force: bool = False) -> list[Path]:
    written: list[Path] = []
    pages = discover_pages(config.images_root, limit=limit)
    for page_ref in pages:
        out_path = pagexml_output_path(config.output_root, page_ref.book_id, page_ref.page_id)
        if out_path.exists() and not force:
            written.append(out_path)
            continue
        consensus_path = consensus_output_path(config.output_root, page_ref.book_id, page_ref.page_id)
        if not consensus_path.exists():
            continue
        page = ConsensusPage.from_json(read_json(consensus_path))
        write_page_xml(out_path, page)
        written.append(out_path)
    return written


def validate_page_xml(path: Path, xsd_path: Path | None = None) -> bool:
    ET.parse(path)
    if xsd_path is None:
        return True
    try:
        from lxml import etree
    except ImportError as exc:
        raise RuntimeError("lxml is required for XSD validation") from exc
    schema = etree.XMLSchema(etree.parse(str(xsd_path)))
    document = etree.parse(str(path))
    return bool(schema.validate(document))
