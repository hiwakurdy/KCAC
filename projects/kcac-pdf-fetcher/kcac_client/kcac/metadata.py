from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from bs4 import BeautifulSoup

from .config import BookSpec

log = logging.getLogger(__name__)

LICENSE_NOTE = (
    "Source: KCAC Archive (https://archive.kcac.org). Downloaded for academic OCR "
    "research. Cite KCAC as data source in any publications."
)


def strip_html(value: Any) -> str:
    """Convert KCAC HTML metadata text into plain text with line breaks.

    Args:
        value: Raw metadata value.

    Returns:
        Plain text value.
    """
    if value is None:
        return ""
    soup = BeautifulSoup(str(value), "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text("\n")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def normalize_meta(raw_meta: dict[str, Any] | list[Any]) -> dict[str, dict[str, Any]]:
    """Normalize KCAC metadata entries by ``machineName``.

    Args:
        raw_meta: Raw ``/api/item/{book_id}/meta`` response.

    Returns:
        Mapping from machineName to metadata entry.
    """
    entries: dict[str, dict[str, Any]] = {}
    if isinstance(raw_meta, dict):
        iterable = raw_meta.values()
    elif isinstance(raw_meta, list):
        iterable = raw_meta
    else:
        return entries

    for item in iterable:
        if not isinstance(item, dict):
            continue
        machine_name = item.get("machineName")
        if isinstance(machine_name, str):
            entries[machine_name] = item
    return entries


def values_for(meta: dict[str, dict[str, Any]], key: str) -> list[Any]:
    """Return all values for a metadata key.

    Args:
        meta: Normalized metadata mapping.
        key: Metadata machineName.

    Returns:
        Value list, or an empty list.
    """
    raw_values = meta.get(key, {}).get("values", [])
    if isinstance(raw_values, list):
        return raw_values
    if raw_values is None:
        return []
    return [raw_values]


def scalar_for(meta: dict[str, dict[str, Any]], key: str) -> Any:
    """Return the first value for a metadata key.

    Args:
        meta: Normalized metadata mapping.
        key: Metadata machineName.

    Returns:
        First value, or None.
    """
    values = values_for(meta, key)
    return values[0] if values else None


def flatten_category(values: list[Any]) -> list[str]:
    """Flatten KCAC category path values.

    Args:
        values: Raw category values.

    Returns:
        Flat category list.
    """
    category: list[str] = []
    for value in values:
        if isinstance(value, list):
            category.extend(str(part) for part in value)
        elif value is not None:
            category.append(str(value))
    return category


def parse_publication_date(raw_value: Any) -> str | None:
    """Parse a KCAC YYYYMMDD publication date.

    Args:
        raw_value: Raw date value.

    Returns:
        ISO date if parseable, otherwise the raw value as a string.
    """
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) == 8 and text.isdigit():
        year = int(text[:4])
        month = int(text[4:6])
        day = int(text[6:8])
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
    return text or None


def timestamp_utc(value: datetime | None) -> str | None:
    """Format a datetime as UTC ISO-8601.

    Args:
        value: Datetime to format.

    Returns:
        Timestamp string or None.
    """
    if value is None:
        return None
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_meta(
    raw_meta: dict[str, Any] | list[Any],
    book_spec: BookSpec,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> dict[str, Any]:
    """Build the cleaned ``metadata.json`` payload.

    Args:
        raw_meta: Raw KCAC metadata response.
        book_spec: Parsed book pages response.
        started_at: Scrape start timestamp.
        finished_at: Scrape finish timestamp.

    Returns:
        Clean metadata dictionary.
    """
    meta = normalize_meta(raw_meta)
    title = strip_html(scalar_for(meta, "title"))
    publication_date_raw = scalar_for(meta, "cf13")
    page_count = scalar_for(meta, "images") or book_spec.total_pages
    page_resolutions = [[page.width, page.height] for page in book_spec.pages]

    return {
        "book_id": book_spec.book_id,
        "source_url": f"https://archive.kcac.org/zoom/{book_spec.book_id}/view",
        "title": title,
        "title_plain": " ".join(title.split()),
        "authors": [str(value) for value in values_for(meta, "creator")],
        "publisher": scalar_for(meta, "publisher"),
        "place_of_publication": scalar_for(meta, "cf18"),
        "publication_date": parse_publication_date(publication_date_raw),
        "publication_date_raw": publication_date_raw,
        "language": scalar_for(meta, "language"),
        "category": flatten_category(values_for(meta, "category")),
        "tags": [str(value) for value in values_for(meta, "tags")],
        "source_collection": scalar_for(meta, "cf17"),
        "page_count": int(page_count) if str(page_count).isdigit() else book_spec.total_pages,
        "page_resolutions": page_resolutions,
        "scrape_started_at": timestamp_utc(started_at),
        "scrape_finished_at": timestamp_utc(finished_at),
        "tile_protocol": "kcac-osd-v1",
        "license_note": LICENSE_NOTE,
    }
