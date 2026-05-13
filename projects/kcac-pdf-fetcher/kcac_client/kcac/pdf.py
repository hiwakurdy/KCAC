from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def assemble_pdf(page_paths: list[Path], out_pdf: Path) -> None:
    """Assemble JPEG page images into a PDF without recompressing.

    Args:
        page_paths: Page JPEG paths.
        out_pdf: Output PDF path.
    """
    sorted_paths = sorted(page_paths)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    import img2pdf

    with out_pdf.open("wb") as handle:
        handle.write(img2pdf.convert([str(path) for path in sorted_paths]))
    log.debug("Wrote PDF %s from %d image(s).", out_pdf, len(sorted_paths))


def pdf_page_count(pdf_path: Path) -> int | None:
    """Return the number of pages in a PDF.

    Args:
        pdf_path: PDF path.

    Returns:
        Page count, or None if the PDF cannot be read.
    """
    try:
        import pypdf

        reader = pypdf.PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception as exc:
        log.error("Could not read PDF %s: %s", pdf_path, exc)
        return None


def verify_pdf(pdf_path: Path, expected_pages: int) -> bool:
    """Verify that a PDF opens and has the expected page count.

    Args:
        pdf_path: PDF path.
        expected_pages: Expected page count.

    Returns:
        True when the PDF page count matches.
    """
    actual_pages = pdf_page_count(pdf_path)
    if actual_pages != expected_pages:
        log.error(
            "PDF page count mismatch for %s: got %s, expected %d",
            pdf_path,
            actual_pages,
            expected_pages,
        )
        return False
    log.info("PDF verified: %s (%d pages)", pdf_path.name, expected_pages)
    return True


def pdf_exists_and_valid(pdf_path: Path, expected_pages: int) -> bool:
    """Return True if a PDF exists and matches the expected page count.

    Args:
        pdf_path: PDF path.
        expected_pages: Expected page count.

    Returns:
        True when the existing PDF is valid.
    """
    return pdf_path.exists() and verify_pdf(pdf_path, expected_pages)


def estimate_dpi(page_paths: list[Path]) -> float | None:
    """Estimate average DPI assuming A4-height pages.

    Args:
        page_paths: Page JPEG paths.

    Returns:
        Estimated DPI, or None if no pages can be opened.
    """
    from PIL import Image

    heights: list[int] = []
    for path in sorted(page_paths)[: min(10, len(page_paths))]:
        try:
            with Image.open(path) as image:
                heights.append(image.size[1])
        except Exception:
            continue
    if not heights:
        return None
    return (sum(heights) / len(heights)) / 11.69
