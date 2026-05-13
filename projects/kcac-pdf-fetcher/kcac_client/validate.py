#!/usr/bin/env python3
"""Validate a KCAC dataset directory and write a CSV report."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image

log = logging.getLogger("kcac.validate")

FIELDNAMES = (
    "book_id",
    "expected_pages",
    "downloaded_pages",
    "missing_pages",
    "min_resolution",
    "max_resolution",
    "avg_resolution_mp",
    "total_size_mb",
    "pdf_exists",
    "pdf_page_count",
    "pdf_matches_json",
    "status",
)


def setup_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument vector.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Validate a KCAC dataset directory and produce a CSV report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", required=True, metavar="DIR", help="Dataset directory.")
    parser.add_argument("--report", default="validation.csv", metavar="CSV", help="Report path.")
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducible spot checks.",
    )
    return parser.parse_args(argv)


def load_metadata(book_dir: Path) -> dict[str, Any] | None:
    """Load book metadata.

    Args:
        book_dir: Book output directory.

    Returns:
        Metadata dictionary, or None if unavailable.
    """
    metadata_path = book_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("[book=%s] Could not load metadata.json: %s", book_dir.name, exc)
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def pdf_page_count(pdf_path: Path) -> int | None:
    """Return the page count for a PDF.

    Args:
        pdf_path: PDF path.

    Returns:
        Page count, or None if unreadable.
    """
    try:
        import pypdf

        reader = pypdf.PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception as exc:
        log.warning("Could not read PDF %s: %s", pdf_path, exc)
        return None


def image_dimensions(path: Path) -> tuple[int, int] | None:
    """Verify and return image dimensions.

    Args:
        path: Image path.

    Returns:
        Image dimensions, or None if corrupt.
    """
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def expected_dimensions(metadata: dict[str, Any] | None) -> dict[int, tuple[int, int]]:
    """Build page-label to expected-dimensions mapping from metadata.

    Args:
        metadata: Parsed metadata dictionary.

    Returns:
        Mapping from one-based page label to dimensions.
    """
    if not metadata:
        return {}
    raw_resolutions = metadata.get("page_resolutions") or []
    mapping: dict[int, tuple[int, int]] = {}
    for index, raw_dims in enumerate(raw_resolutions, start=1):
        if (
            isinstance(raw_dims, list)
            and len(raw_dims) == 2
            and all(isinstance(value, int) for value in raw_dims)
        ):
            mapping[index] = (raw_dims[0], raw_dims[1])
    return mapping


def page_label(path: Path) -> int | None:
    """Extract the page label from ``page_NNNN.jpg``.

    Args:
        path: Page image path.

    Returns:
        Page label, or None.
    """
    try:
        return int(path.stem.replace("page_", ""))
    except ValueError:
        return None


def spot_check_pages(
    page_files: list[Path],
    dims_by_label: dict[int, tuple[int, int]],
    rng: random.Random,
    sample_size: int = 3,
) -> list[str]:
    """Spot-check random pages for integrity and dimensions.

    Args:
        page_files: Page JPEG paths.
        dims_by_label: Expected dimensions by page label.
        rng: Random generator.
        sample_size: Number of pages to sample.

    Returns:
        Warning strings.
    """
    warnings: list[str] = []
    for path in rng.sample(page_files, min(sample_size, len(page_files))):
        dims = image_dimensions(path)
        if dims is None:
            warnings.append(f"CORRUPT:{path.name}")
            continue
        label = page_label(path)
        if label is not None and label in dims_by_label and dims != dims_by_label[label]:
            warnings.append(f"DIM_MISMATCH:{path.name}:got={dims}:expected={dims_by_label[label]}")
    return warnings


def validate_book(book_dir: Path, rng: random.Random) -> dict[str, Any]:
    """Validate one book directory.

    Args:
        book_dir: Book output directory.
        rng: Random generator for spot checks.

    Returns:
        CSV row dictionary.
    """
    book_id = book_dir.name
    metadata = load_metadata(book_dir)
    expected_pages = int(metadata.get("page_count", 0)) if metadata else 0
    dims_by_label = expected_dimensions(metadata)

    pages_dir = book_dir / "pages"
    page_files = sorted(pages_dir.glob("page_*.jpg")) if pages_dir.exists() else []
    downloaded_pages = len(page_files)
    labels = {label for path in page_files if (label := page_label(path)) is not None}
    missing_pages = sorted(set(range(1, expected_pages + 1)) - labels) if expected_pages else []

    valid_dims: list[tuple[int, int]] = []
    corrupt_pages: list[str] = []
    dimension_mismatches: list[str] = []
    for path in page_files:
        dims = image_dimensions(path)
        if dims is None:
            corrupt_pages.append(path.name)
            continue
        valid_dims.append(dims)
        label = page_label(path)
        if label is not None and label in dims_by_label and dims != dims_by_label[label]:
            dimension_mismatches.append(path.name)

    spot_warnings = spot_check_pages(page_files, dims_by_label, rng)
    for warning in spot_warnings:
        log.warning("[book=%s] spot check: %s", book_id, warning)

    total_size_mb = sum(path.stat().st_size for path in page_files) / 1_048_576
    min_resolution = min(valid_dims, key=lambda dims: dims[0] * dims[1]) if valid_dims else None
    max_resolution = max(valid_dims, key=lambda dims: dims[0] * dims[1]) if valid_dims else None
    mp_values = [width * height / 1_000_000 for width, height in valid_dims]
    avg_resolution_mp = sum(mp_values) / len(mp_values) if mp_values else 0.0

    pdf_path = book_dir / f"{book_id}.pdf"
    pdf_exists = pdf_path.exists()
    pdf_count = pdf_page_count(pdf_path) if pdf_exists else None
    pdf_matches_json = bool(expected_pages and pdf_count == expected_pages)

    if downloaded_pages == 0:
        status = "failed"
    elif pdf_exists and not pdf_matches_json:
        status = "pdf_mismatch"
    elif missing_pages or corrupt_pages or dimension_mismatches or spot_warnings or not pdf_exists:
        status = "partial"
    else:
        status = "complete"

    return {
        "book_id": book_id,
        "expected_pages": expected_pages or "",
        "downloaded_pages": downloaded_pages,
        "missing_pages": ";".join(str(page) for page in missing_pages),
        "min_resolution": format_resolution(min_resolution),
        "max_resolution": format_resolution(max_resolution),
        "avg_resolution_mp": f"{avg_resolution_mp:.2f}",
        "total_size_mb": f"{total_size_mb:.2f}",
        "pdf_exists": pdf_exists,
        "pdf_page_count": pdf_count if pdf_count is not None else "",
        "pdf_matches_json": pdf_matches_json,
        "status": status,
    }


def format_resolution(dims: tuple[int, int] | None) -> str:
    """Format dimensions as WIDTHxHEIGHT.

    Args:
        dims: Optional dimensions tuple.

    Returns:
        Formatted string.
    """
    if dims is None:
        return ""
    return f"{dims[0]}x{dims[1]}"


def write_report(rows: list[dict[str, Any]], report_path: Path) -> None:
    """Write validation rows to CSV.

    Args:
        rows: CSV rows.
        report_path: Destination path.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    """Run the validator CLI.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit code.
    """
    setup_logging()
    args = parse_args(argv)
    output_dir = Path(args.output)
    if not output_dir.is_dir():
        log.error("Dataset directory not found: %s", output_dir)
        return 3

    book_dirs = sorted(path for path in output_dir.iterdir() if path.is_dir() and path.name.isdigit())
    if not book_dirs:
        log.error("No numeric book directories found in %s", output_dir)
        return 3

    rng = random.Random(args.seed)
    rows = [validate_book(book_dir, rng) for book_dir in book_dirs]
    report_path = Path(args.report)
    write_report(rows, report_path)
    log.info("Wrote validation report: %s", report_path)

    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1

    print("\nValidation summary:")
    for status, count in sorted(counts.items()):
        print(f"  {status:<12} {count}")

    return 0 if all(row["status"] == "complete" for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
