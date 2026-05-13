from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from playwright.sync_api import Page, Response

try:
    from playwright.sync_api import Error as PlaywrightError
except ModuleNotFoundError:
    PlaywrightError = Exception


IIIF_CONTEXT_RE = re.compile(r"iiif\.io/api/(?:presentation|image)/[23]", re.I)
MANIFEST_URL_RE = re.compile(r"(?:^|/)(?:manifest(?:\.json)?|iiif/manifest)(?:[/?#]|$)", re.I)
INFO_JSON_RE = re.compile(r"/info\.json(?:[?#].*)?$", re.I)
IIIF_IMAGE_URL_RE = re.compile(
    r"/(?:full|square|pct:[^/]+|[0-9]+,[0-9]+,[0-9]+,[0-9]+)"
    r"/(?:full|max|pct:[^/]+|!?[0-9]*,?[0-9]*)"
    r"/(?:[0-9.]+|![0-9.]+)"
    r"/(?:default|color|gray|bitonal)\."
    r"(?:jpg|jpeg|png|tif|tiff|webp|jp2)(?:[?#].*)?$",
    re.I,
)


@dataclass(frozen=True)
class DiscoverConfig:
    """Runtime configuration for the IIIF discovery browser session."""

    viewer_url: str
    output_log: Path
    headless: bool = True
    timeout_ms: int = 45_000
    settle_seconds: float = 5.0
    zoom_steps: int = 2
    user_agent: str = (
        "Kurdish-OCR-Research/1.0 "
        "(PhD dissertation; contact: replace-with-your-email)"
    )


@dataclass
class NetworkEntry:
    """Serializable record for one observed request/response pair."""

    url: str
    method: str
    resource_type: str
    status: int | None = None
    content_type: str | None = None
    matched_signals: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class DiscoveryResult:
    """Protocol evidence collected from the browser session."""

    manifest_urls: list[str] = field(default_factory=list)
    manifest_context_urls: list[str] = field(default_factory=list)
    info_json_urls: list[str] = field(default_factory=list)
    image_api_urls: list[str] = field(default_factory=list)
    image_service_urls: list[str] = field(default_factory=list)
    network_entries: list[NetworkEntry] = field(default_factory=list)


def parse_args() -> DiscoverConfig:
    """Parse command-line arguments.

    Returns:
        A discovery configuration object.
    """
    parser = argparse.ArgumentParser(
        description="Discover IIIF Presentation/Image API endpoints from a viewer URL."
    )
    parser.add_argument(
        "--viewer-url",
        required=True,
        help="Book viewer URL, for example https://archive.kcac.org/zoom/399/view",
    )
    parser.add_argument(
        "--output-log",
        type=Path,
        default=None,
        help="JSON network log path. Defaults to iiif_discover_<book_id>.json.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=45_000,
        help="Page load timeout in milliseconds.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=5.0,
        help="Seconds to wait after load and after viewer interactions.",
    )
    parser.add_argument(
        "--zoom-steps",
        type=int,
        default=2,
        help="Number of zoom interactions to attempt after page load.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run Chromium headed for debugging.",
    )
    parser.add_argument(
        "--user-agent",
        default=DiscoverConfig.user_agent,
        help="Identifying User-Agent to send with browser requests.",
    )
    args = parser.parse_args()
    output_log = args.output_log or default_log_path(args.viewer_url)
    return DiscoverConfig(
        viewer_url=args.viewer_url,
        output_log=output_log,
        headless=not args.headed,
        timeout_ms=args.timeout_ms,
        settle_seconds=args.settle_seconds,
        zoom_steps=args.zoom_steps,
        user_agent=args.user_agent,
    )


def default_log_path(viewer_url: str) -> Path:
    """Build the default network log path for a viewer URL.

    Args:
        viewer_url: The viewer URL being inspected.

    Returns:
        A path like ``iiif_discover_399.json`` when a book id is visible.
    """
    match = re.search(r"/zoom/([^/]+)/view", viewer_url)
    suffix = match.group(1) if match else "network"
    safe_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", suffix)
    return Path(f"iiif_discover_{safe_suffix}.json")


def configure_logging() -> None:
    """Configure console logging for the discovery run."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def add_unique(items: list[str], value: str) -> None:
    """Append a string to a list if it is not already present.

    Args:
        items: Destination list.
        value: String value to append.
    """
    if value and value not in items:
        items.append(value)


def signals_for_url(url: str) -> list[str]:
    """Return IIIF-related detection signals for a URL.

    Args:
        url: Request or response URL.

    Returns:
        A list of matched signal names.
    """
    signals: list[str] = []
    if looks_like_manifest_url(url):
        signals.append("manifest-url")
    if looks_like_info_json_url(url):
        signals.append("info-json")
    if looks_like_iiif_image_url(url):
        signals.append("iiif-image-url")
    return signals


def looks_like_manifest_url(url: str) -> bool:
    """Check whether a URL resembles an IIIF Presentation manifest URL.

    Args:
        url: URL to inspect.

    Returns:
        True when the URL path matches common manifest patterns.
    """
    return bool(MANIFEST_URL_RE.search(urlsplit(url).path))


def looks_like_info_json_url(url: str) -> bool:
    """Check whether a URL resembles an IIIF Image API info.json endpoint.

    Args:
        url: URL to inspect.

    Returns:
        True when the URL path ends with ``/info.json``.
    """
    return bool(INFO_JSON_RE.search(urlsplit(url).path))


def looks_like_iiif_image_url(url: str) -> bool:
    """Check whether a URL matches the IIIF Image API request template.

    Args:
        url: URL to inspect.

    Returns:
        True when the path contains the IIIF region/size/rotation/quality
        segments used for image delivery.
    """
    return bool(IIIF_IMAGE_URL_RE.search(urlsplit(url).path))


def service_from_info_json(url: str) -> str:
    """Extract the Image API service id from an ``info.json`` URL.

    Args:
        url: IIIF ``info.json`` URL.

    Returns:
        The URL with the trailing ``/info.json`` segment removed.
    """
    parts = urlsplit(url)
    path = re.sub(r"/info\.json$", "", parts.path, flags=re.I)
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def service_from_image_url(url: str) -> str | None:
    """Extract the Image API service id from a full IIIF image URL.

    Args:
        url: URL matching the IIIF Image API delivery template.

    Returns:
        The image service id, or None if the URL is too short.
    """
    parts = urlsplit(url)
    segments = [segment for segment in parts.path.split("/") if segment]
    if len(segments) < 5:
        return None
    service_segments = segments[:-4]
    service_path = "/" + "/".join(service_segments)
    return urlunsplit((parts.scheme, parts.netloc, service_path, "", ""))


def inspect_json_response(url: str, text: str, result: DiscoveryResult) -> None:
    """Inspect JSON response text for IIIF Presentation or Image API evidence.

    Args:
        url: Response URL.
        text: JSON response body.
        result: Mutable discovery result.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logging.debug("Could not parse JSON response from %s", url)
        return

    if payload_contains_iiif_presentation_context(payload):
        add_unique(result.manifest_context_urls, url)
        add_unique(result.manifest_urls, url)

    if payload_contains_iiif_image_context(payload):
        add_unique(result.info_json_urls, url)
        add_unique(result.image_service_urls, service_from_info_json(url))


def payload_contains_iiif_presentation_context(payload: Any) -> bool:
    """Check whether a JSON payload declares an IIIF Presentation context.

    Args:
        payload: Parsed JSON payload.

    Returns:
        True if the payload looks like an IIIF Presentation resource.
    """
    contexts = extract_context_values(payload)
    return any("iiif.io/api/presentation" in context for context in contexts)


def payload_contains_iiif_image_context(payload: Any) -> bool:
    """Check whether a JSON payload declares an IIIF Image API context.

    Args:
        payload: Parsed JSON payload.

    Returns:
        True if the payload looks like an IIIF Image API resource.
    """
    contexts = extract_context_values(payload)
    if any("iiif.io/api/image" in context for context in contexts):
        return True
    return isinstance(payload, dict) and "tiles" in payload and "width" in payload


def extract_context_values(payload: Any) -> list[str]:
    """Extract IIIF context strings from a JSON payload.

    Args:
        payload: Parsed JSON payload.

    Returns:
        Context values as strings.
    """
    if not isinstance(payload, dict):
        return []
    raw_context = payload.get("@context", payload.get("context"))
    if isinstance(raw_context, str):
        return [raw_context]
    if isinstance(raw_context, list):
        return [item for item in raw_context if isinstance(item, str)]
    return []


def maybe_read_response_body(response: Response) -> str | None:
    """Read a small likely-JSON response body from Playwright.

    Args:
        response: Playwright response object.

    Returns:
        Response text when it is safe and useful to inspect, otherwise None.
    """
    headers = response.headers
    content_type = headers.get("content-type", "").lower()
    url = response.url
    likely_json = (
        "json" in content_type
        or looks_like_manifest_url(url)
        or looks_like_info_json_url(url)
    )
    if not likely_json:
        return None

    length_header = headers.get("content-length")
    if length_header and length_header.isdigit() and int(length_header) > 10_000_000:
        logging.debug("Skipping large JSON-like response body from %s", url)
        return None

    try:
        return response.text()
    except PlaywrightError as exc:
        logging.debug("Could not read response body from %s: %s", url, exc)
        return None


def register_response(result: DiscoveryResult, response: Response) -> None:
    """Record a Playwright response and update IIIF discovery evidence.

    Args:
        result: Mutable discovery result.
        response: Playwright response object.
    """
    request = response.request
    url = response.url
    signals = signals_for_url(url)
    content_type = response.headers.get("content-type")
    entry = NetworkEntry(
        url=url,
        method=request.method,
        resource_type=request.resource_type,
        status=response.status,
        content_type=content_type,
        matched_signals=signals,
    )
    result.network_entries.append(entry)

    if "manifest-url" in signals:
        add_unique(result.manifest_urls, url)
    if "info-json" in signals:
        add_unique(result.info_json_urls, url)
        add_unique(result.image_service_urls, service_from_info_json(url))
    if "iiif-image-url" in signals:
        add_unique(result.image_api_urls, url)
        service_url = service_from_image_url(url)
        if service_url:
            add_unique(result.image_service_urls, service_url)

    body = maybe_read_response_body(response)
    if body and IIIF_CONTEXT_RE.search(body[:10_000]):
        inspect_json_response(url, body, result)


def register_request_failure(result: DiscoveryResult, request: Any) -> None:
    """Record a failed Playwright request in the network log.

    Args:
        result: Mutable discovery result.
        request: Playwright request object.
    """
    failure = request.failure
    result.network_entries.append(
        NetworkEntry(
            url=request.url,
            method=request.method,
            resource_type=request.resource_type,
            matched_signals=signals_for_url(request.url),
            error=failure or "request failed",
        )
    )


def interact_with_viewer(page: Page, zoom_steps: int, settle_seconds: float) -> None:
    """Try common viewer interactions to trigger IIIF network requests.

    Args:
        page: Playwright page.
        zoom_steps: Number of zoom interactions to attempt.
        settle_seconds: Seconds to wait after interaction.
    """
    for _ in range(max(zoom_steps, 0)):
        clicked = click_zoom_button(page)
        if not clicked:
            page.mouse.wheel(0, -900)
        time.sleep(0.5)
    time.sleep(settle_seconds)


def click_zoom_button(page: Page) -> bool:
    """Click a likely zoom-in control if one is present.

    Args:
        page: Playwright page.

    Returns:
        True if a control was clicked, otherwise False.
    """
    selectors = [
        "button[aria-label*='Zoom in' i]",
        "button[title*='Zoom in' i]",
        ".openseadragon-container button[title*='Zoom in' i]",
        ".zoom-in",
        ".zoomIn",
        "[class*='zoom'][class*='in']",
    ]
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.count() and locator.is_visible(timeout=500):
                locator.click(timeout=1_000)
                return True
        except PlaywrightError:
            continue
    return False


def discover_iiif(config: DiscoverConfig) -> DiscoveryResult:
    """Run a browser session and discover IIIF endpoints.

    Args:
        config: Discovery configuration.

    Returns:
        Collected discovery evidence.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Playwright is not installed. Install dependencies with "
            "`pip install playwright` and then run `playwright install chromium`."
        ) from exc

    result = DiscoveryResult()
    logging.info("Loading viewer: %s", config.viewer_url)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=config.headless)
        context = browser.new_context(user_agent=config.user_agent)
        page = context.new_page()
        page.on("response", lambda response: register_response(result, response))
        page.on("requestfailed", lambda request: register_request_failure(result, request))

        try:
            page.goto(
                config.viewer_url,
                wait_until="networkidle",
                timeout=config.timeout_ms,
            )
        except PlaywrightError as exc:
            logging.warning("Initial network-idle load did not complete: %s", exc)
            page.goto(config.viewer_url, wait_until="domcontentloaded")

        time.sleep(config.settle_seconds)
        interact_with_viewer(page, config.zoom_steps, config.settle_seconds)
        context.close()
        browser.close()
    return result


def write_network_log(config: DiscoverConfig, result: DiscoveryResult) -> None:
    """Write full discovery evidence to JSON.

    Args:
        config: Discovery configuration.
        result: Discovery evidence to serialize.
    """
    payload = {
        "viewer_url": config.viewer_url,
        "summary": {
            "manifest_urls": result.manifest_urls,
            "manifest_context_urls": result.manifest_context_urls,
            "info_json_urls": result.info_json_urls,
            "image_api_urls": result.image_api_urls,
            "image_service_urls": result.image_service_urls,
        },
        "network": [asdict(entry) for entry in result.network_entries],
    }
    config.output_log.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logging.info("Wrote network log: %s", config.output_log)


def print_verdict(result: DiscoveryResult) -> int:
    """Print a human-readable IIIF discovery verdict.

    Args:
        result: Discovery evidence.

    Returns:
        Process exit code.
    """
    print()
    print("IIIF discovery verdict")
    print("======================")

    if result.manifest_urls:
        print("Status: IIIF Presentation API manifest detected")
        print(f"Manifest URL: {result.manifest_urls[0]}")
        if len(result.manifest_urls) > 1:
            print("Additional manifest candidates:")
            for url in result.manifest_urls[1:]:
                print(f"  - {url}")
        if result.image_service_urls:
            print("Image service samples:")
            for url in result.image_service_urls[:5]:
                print(f"  - {url}")
        return 0

    if result.info_json_urls or result.image_api_urls:
        print("Status: IIIF Image API detected, but no Presentation manifest was captured")
        if result.info_json_urls:
            print("info.json samples:")
            for url in result.info_json_urls[:5]:
                print(f"  - {url}")
        if result.image_api_urls:
            print("Image request samples:")
            for url in result.image_api_urls[:5]:
                print(f"  - {url}")
        if result.image_service_urls:
            print("Image service samples:")
            for url in result.image_service_urls[:5]:
                print(f"  - {url}")
        print()
        print(
            "Recommendation: inspect the JSON log for a manifest URL embedded in "
            "page scripts, or run again with --headed to interact manually."
        )
        return 2

    print("Status: no IIIF Presentation or Image API endpoint detected")
    print(
        "Recommendation: inspect the JSON network log. This discoverer is IIIF-only "
        "and will not attempt DeepZoom, Zoomify, authentication, or bypass logic."
    )
    return 1


def main() -> int:
    """Run the IIIF discovery command-line program.

    Returns:
        Process exit code.
    """
    configure_logging()
    config = parse_args()
    try:
        result = discover_iiif(config)
        write_network_log(config, result)
        return print_verdict(result)
    except KeyboardInterrupt:
        logging.error("Interrupted")
        return 130
    except PlaywrightError as exc:
        logging.error("Playwright failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
