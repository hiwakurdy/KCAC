from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import requests

from .api import post_json
from .config import Config, PageSpec, RequestState

log = logging.getLogger(__name__)


def extract_clip_text(
    session: requests.Session,
    cfg: Config,
    state: RequestState,
    book_id: int,
    page: PageSpec,
    margin: int = 0,
) -> list[dict[str, Any]]:
    """Extract text for a full-page KCAC clip zone.

    Args:
        session: Shared HTTP session.
        cfg: Runtime configuration.
        state: Mutable per-run request state.
        book_id: KCAC item id.
        page: Page specification.
        margin: Optional inset from each page edge in pixels.

    Returns:
        Raw clip API response as a list of zone dictionaries.
    """
    safe_margin = max(0, margin)
    width = max(1, page.width - (2 * safe_margin))
    height = max(1, page.height - (2 * safe_margin))
    payload = {
        "id": f"c-{book_id}-{page.label:04d}",
        "type": "text",
        "pageId": page.id,
        "rect": {
            "x": safe_margin,
            "y": safe_margin,
            "width": width,
            "height": height,
        },
    }
    url = f"{cfg.base_url.rstrip('/')}/api/item/{book_id}/clip/{page.id}/find"
    log.debug("POST clip text book=%d page=%d page_id=%d url=%s", book_id, page.label, page.id, url)
    response = post_json(session, cfg, state, url, payload)
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    if isinstance(response, dict):
        return [response]
    return []


def text_from_zones(zones: list[dict[str, Any]]) -> str:
    """Join text fields from KCAC clip zones.

    Args:
        zones: Raw zone dictionaries.

    Returns:
        Combined text.
    """
    parts: list[str] = []
    for zone in zones:
        text = zone.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return "\n".join(parts)


def normalize_text_lines(text: str) -> list[str]:
    """Normalize OCR text so each saved line occupies one output line.

    Args:
        text: Raw OCR text from KCAC.

    Returns:
        Clean non-empty lines.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for line in normalized.split("\n"):
        collapsed = re.sub(r"[ \t\f\v]+", " ", line).strip()
        if collapsed:
            lines.append(collapsed)
    return lines


def write_text_outputs(
    book_dir: Path,
    page: PageSpec,
    zones: list[dict[str, Any]],
    write_raw: bool = True,
) -> tuple[Path, list[str]]:
    """Write per-page OCR text and optional raw JSON.

    Args:
        book_dir: Book output directory.
        page: Page specification.
        zones: Raw clip API zones.
        write_raw: Whether to write raw JSON under ``text_raw``.

    Returns:
        Text output path and normalized lines.
    """
    text_dir = book_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    lines = normalize_text_lines(text_from_zones(zones))
    text_path = text_dir / f"page_{page.label:04d}.txt"
    text_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if write_raw:
        raw_dir = book_dir / "text_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"page_{page.label:04d}.json"
        raw_path.write_text(json.dumps(zones, indent=2, ensure_ascii=False), encoding="utf-8")

    return text_path, lines


def text_exists(path: Path) -> bool:
    """Return True when a text file exists and can be read.

    Args:
        path: Text file path.

    Returns:
        True if the file is readable.
    """
    if not path.exists():
        return False
    try:
        path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    return True
