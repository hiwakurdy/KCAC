from __future__ import annotations

import json
import logging
import time
import urllib.robotparser
from typing import Any

import requests

from .config import BookSpec, Config, PageSpec, RequestState, TileSpec

log = logging.getLogger(__name__)

RATE_BACKOFF_SECONDS = (10, 30, 90, 270)
CONNECTION_BACKOFF_SECONDS = (5, 15, 45)


class KcacError(Exception):
    """Base exception for recoverable KCAC client errors."""


class SetupError(KcacError):
    """Raised when the run cannot start safely."""


class AuthenticationRequired(KcacError):
    """Raised when a public endpoint starts requiring authentication."""


class RateLimitAbort(KcacError):
    """Raised after repeated 429/503 responses."""


class PermanentFetchError(KcacError):
    """Raised when a URL cannot be fetched after all retries."""


def build_session(cfg: Config) -> requests.Session:
    """Create a configured requests session.

    Args:
        cfg: Runtime configuration.

    Returns:
        A requests session with the identifying User-Agent installed.
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": cfg.user_agent,
            "Accept": "application/json, image/jpeg, */*",
        }
    )
    return session


def check_robots_txt(session: requests.Session, cfg: Config) -> None:
    """Check robots.txt and abort if the KCAC API is disallowed.

    Args:
        session: Shared HTTP session.
        cfg: Runtime configuration.

    Raises:
        SetupError: If robots.txt explicitly disallows ``/api/``.
    """
    robots_url = f"{cfg.base_url.rstrip('/')}/robots.txt"
    api_url = f"{cfg.base_url.rstrip('/')}/api/"
    log.info("Checking robots.txt: %s", robots_url)
    try:
        response = session.get(robots_url, timeout=30)
        if response.status_code in (401, 403):
            raise SetupError(
                f"robots.txt returned HTTP {response.status_code}. "
                "Contact KCAC before running the scraper."
            )
        response.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Could not fetch robots.txt (%s); proceeding cautiously.", exc)
        return

    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    parser.parse(response.text.splitlines())
    if not parser.can_fetch(cfg.user_agent, api_url):
        raise SetupError(
            "robots.txt disallows /api/. Stop here and contact KCAC for permission."
        )
    log.info("robots.txt permits /api/ for this user agent.")


def fetch_meta(
    session: requests.Session,
    cfg: Config,
    state: RequestState,
    book_id: int,
) -> dict[str, Any]:
    """Fetch raw book metadata.

    Args:
        session: Shared HTTP session.
        cfg: Runtime configuration.
        state: Mutable per-run request state.
        book_id: KCAC item id.

    Returns:
        Raw metadata object.
    """
    url = f"{cfg.base_url.rstrip('/')}/api/item/{book_id}/meta"
    log.debug("GET %s", url)
    payload = fetch_json(session, cfg, state, url)
    if not isinstance(payload, dict):
        raise PermanentFetchError(f"Metadata for book {book_id} was not a JSON object.")
    return payload


def fetch_pages(
    session: requests.Session,
    cfg: Config,
    state: RequestState,
    book_id: int,
) -> dict[str, Any]:
    """Fetch the raw KCAC page list for a book.

    Args:
        session: Shared HTTP session.
        cfg: Runtime configuration.
        state: Mutable per-run request state.
        book_id: KCAC item id.

    Returns:
        Raw page-list object.
    """
    url = f"{cfg.base_url.rstrip('/')}/api/item/{book_id}/pages"
    log.debug("GET %s", url)
    payload = fetch_json(session, cfg, state, url)
    if not isinstance(payload, dict):
        raise PermanentFetchError(f"Pages for book {book_id} were not a JSON object.")
    return payload


def fetch_tile(
    session: requests.Session,
    cfg: Config,
    state: RequestState,
    url: str,
) -> bytes:
    """Fetch one JPEG tile.

    Args:
        session: Shared HTTP session.
        cfg: Runtime configuration.
        state: Mutable per-run request state.
        url: Tile URL.

    Returns:
        Tile bytes.
    """
    data = fetch_bytes(session, cfg, state, url)
    if len(data) < 1024:
        log.warning("Tile at %s is suspiciously small (%d bytes).", url, len(data))
    return data


def fetch_thumbnail(
    session: requests.Session,
    cfg: Config,
    state: RequestState,
    page_id: int,
) -> bytes:
    """Fetch a page thumbnail.

    Args:
        session: Shared HTTP session.
        cfg: Runtime configuration.
        state: Mutable per-run request state.
        page_id: KCAC page id.

    Returns:
        JPEG thumbnail bytes.
    """
    url = f"{cfg.base_url.rstrip('/')}/api/page/{page_id}/thumbnail"
    log.debug("GET %s", url)
    return fetch_bytes(session, cfg, state, url)


def fetch_json(
    session: requests.Session,
    cfg: Config,
    state: RequestState,
    url: str,
) -> Any:
    """Fetch and decode JSON from a URL.

    Args:
        session: Shared HTTP session.
        cfg: Runtime configuration.
        state: Mutable per-run request state.
        url: JSON endpoint.

    Returns:
        Decoded JSON payload.
    """
    data = fetch_bytes(session, cfg, state, url)
    return decode_json_bytes(data, url)


def post_json(
    session: requests.Session,
    cfg: Config,
    state: RequestState,
    url: str,
    payload: Any,
) -> Any:
    """POST JSON and decode the JSON response.

    Args:
        session: Shared HTTP session.
        cfg: Runtime configuration.
        state: Mutable per-run request state.
        url: JSON endpoint.
        payload: JSON-serializable request body.

    Returns:
        Decoded JSON payload.
    """
    data = request_bytes(session, cfg, state, "POST", url, json_payload=payload)
    return decode_json_bytes(data, url)


def decode_json_bytes(data: bytes, url: str) -> Any:
    """Decode a UTF-8 JSON response body.

    Args:
        data: Response body bytes.
        url: Source URL for error messages.

    Returns:
        Decoded JSON payload.
    """
    try:
        return json.loads(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise PermanentFetchError(f"Response from {url} was not UTF-8 JSON.") from exc
    except json.JSONDecodeError as exc:
        raise PermanentFetchError(f"Response from {url} was not valid JSON.") from exc


def fetch_bytes(
    session: requests.Session,
    cfg: Config,
    state: RequestState,
    url: str,
) -> bytes:
    """Fetch bytes from a URL with KCAC politeness and retry rules.

    Args:
        session: Shared HTTP session.
        cfg: Runtime configuration.
        state: Mutable per-run request state.
        url: URL to fetch.

    Returns:
        Response body bytes.

    Raises:
        AuthenticationRequired: If the server returns 401 or 403.
        RateLimitAbort: If the server rate-limits five times consecutively.
        PermanentFetchError: If retries are exhausted.
    """
    return request_bytes(session, cfg, state, "GET", url)


def request_bytes(
    session: requests.Session,
    cfg: Config,
    state: RequestState,
    method: str,
    url: str,
    json_payload: Any | None = None,
) -> bytes:
    """Make one HTTP request with KCAC politeness and retry rules.

    Args:
        session: Shared HTTP session.
        cfg: Runtime configuration.
        state: Mutable per-run request state.
        method: HTTP method.
        url: URL to request.
        json_payload: Optional JSON request body.

    Returns:
        Response body bytes.
    """
    connection_failures = 0
    while True:
        try:
            response = session.request(method, url, timeout=30, json=json_payload)
        except (requests.ConnectionError, requests.Timeout) as exc:
            if connection_failures >= cfg.max_retries:
                raise PermanentFetchError(
                    f"Connection error fetching {url} after {cfg.max_retries} retries: {exc}"
                ) from exc
            wait = CONNECTION_BACKOFF_SECONDS[
                min(connection_failures, len(CONNECTION_BACKOFF_SECONDS) - 1)
            ]
            connection_failures += 1
            log.warning(
                "Connection error at %s: %s. Sleeping %ds (%d/%d).",
                url,
                exc,
                wait,
                connection_failures,
                cfg.max_retries,
            )
            time.sleep(wait)
            continue

        if response.status_code in (401, 403):
            raise AuthenticationRequired(
                f"Server returned HTTP {response.status_code} for {url}. "
                "Do not attempt to bypass authentication; contact KCAC for access."
            )

        if response.status_code in (429, 503):
            state.consecutive_rate_limits += 1
            if state.consecutive_rate_limits >= 5:
                raise RateLimitAbort(
                    "Server returned HTTP 429/503 five times consecutively. "
                    "Stopping the run to avoid hammering KCAC."
                )
            wait = RATE_BACKOFF_SECONDS[
                min(state.consecutive_rate_limits - 1, len(RATE_BACKOFF_SECONDS) - 1)
            ]
            log.warning(
                "HTTP %d at %s. Sleeping %ds after rate-limit response %d/5.",
                response.status_code,
                url,
                wait,
                state.consecutive_rate_limits,
            )
            time.sleep(wait)
            continue

        state.consecutive_rate_limits = 0

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise PermanentFetchError(
                f"HTTP {response.status_code} fetching {url}: {response.text[:200]}"
            ) from exc

        return response.content


def parse_book_spec(book_id: int, pages_data: dict[str, Any]) -> BookSpec:
    """Parse the KCAC pages response into typed page specs.

    Args:
        book_id: KCAC item id.
        pages_data: Raw ``/api/item/{book_id}/pages`` response.

    Returns:
        Typed book specification.

    Raises:
        PermanentFetchError: If required fields are missing.
    """
    try:
        raw_pages = pages_data["pages"]
        raw_total = pages_data["attributes"]["total"]
    except KeyError as exc:
        raise PermanentFetchError(f"Pages response for book {book_id} is incomplete.") from exc

    if not isinstance(raw_pages, list):
        raise PermanentFetchError(f"Pages response for book {book_id} does not contain a list.")

    pages: list[PageSpec] = []
    for index, raw_page in enumerate(raw_pages, start=1):
        try:
            attributes = raw_page["attributes"]
            tile = attributes["tile"]
            tilemap = raw_page["tilemap"]
            label = attributes.get("label", index)
            pages.append(
                PageSpec(
                    id=int(attributes["id"]),
                    width=int(attributes["width"]),
                    height=int(attributes["height"]),
                    tile=TileSpec(width=int(tile["width"]), height=int(tile["height"])),
                    levels=int(attributes["levels"]),
                    label=int(label),
                    tile_uri_template=str(tilemap["uri"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PermanentFetchError(
                f"Page entry {index} for book {book_id} is incomplete."
            ) from exc

    return BookSpec(book_id=book_id, total_pages=int(raw_total), pages=pages)
