#!/usr/bin/env python3
"""Download full-resolution KCAC books and assemble research PDFs."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from kcac.api import (
    AuthenticationRequired,
    PermanentFetchError,
    RateLimitAbort,
    SetupError,
    build_session,
    check_robots_txt,
    fetch_meta,
    fetch_pages,
    fetch_thumbnail,
    parse_book_spec,
)
from kcac.config import BookSpec, Config, RequestState
from kcac.metadata import parse_meta
from kcac.pdf import assemble_pdf, estimate_dpi, pdf_exists_and_valid, verify_pdf
from kcac.stitch import (
    StitchError,
    page_exists_and_valid,
    save_page_jpeg,
    stitch_page,
    verify_page,
)

log = logging.getLogger("kcac.fetch")


@dataclass(frozen=True)
class BookResult:
    """Summary of one book run."""

    book_id: int
    expected_pages: int
    downloaded_pages: int
    size_mb: float
    status: str


class ProgressTracker:
    """JSON-backed progress tracker updated after every page."""

    def __init__(self, path: Path) -> None:
        """Load an existing progress file.

        Args:
            path: Path to ``progress.json``.
        """
        self.path = path
        self._data: dict[str, Any] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("Could not parse %s; starting a new progress file.", path)

    def start_book(self, book_id: int, expected_pages: int) -> None:
        """Mark a book as in progress.

        Args:
            book_id: KCAC item id.
            expected_pages: Expected page count.
        """
        key = str(book_id)
        entry = self._data.setdefault(key, {})
        entry["status"] = "in_progress"
        entry["expected_pages"] = expected_pages
        entry.setdefault("completed_pages", [])
        entry.setdefault("failed_pages", [])
        entry["updated_at"] = utc_now()
        self._save()

    def mark_page_done(self, book_id: int, page_label: int, suspect: bool = False) -> None:
        """Record a completed page.

        Args:
            book_id: KCAC item id.
            page_label: Human-readable page label.
            suspect: Whether validation produced a warning.
        """
        entry = self._entry(book_id)
        completed = set(entry.setdefault("completed_pages", []))
        failed = set(entry.setdefault("failed_pages", []))
        completed.add(page_label)
        failed.discard(page_label)
        entry["completed_pages"] = sorted(completed)
        entry["failed_pages"] = sorted(failed)
        if suspect:
            suspect_pages = set(entry.setdefault("suspect_pages", []))
            suspect_pages.add(page_label)
            entry["suspect_pages"] = sorted(suspect_pages)
        entry["status"] = "in_progress"
        entry["updated_at"] = utc_now()
        self._save()

    def mark_page_failed(self, book_id: int, page_label: int, reason: str) -> None:
        """Record a failed page.

        Args:
            book_id: KCAC item id.
            page_label: Human-readable page label.
            reason: Failure detail.
        """
        entry = self._entry(book_id)
        failed = set(entry.setdefault("failed_pages", []))
        failed.add(page_label)
        entry["failed_pages"] = sorted(failed)
        failures = entry.setdefault("page_errors", {})
        failures[str(page_label)] = reason
        entry["status"] = "in_progress"
        entry["updated_at"] = utc_now()
        self._save()

    def complete_book(self, book_id: int) -> None:
        """Mark a book complete.

        Args:
            book_id: KCAC item id.
        """
        entry = self._entry(book_id)
        entry["status"] = "complete"
        entry["updated_at"] = utc_now()
        self._save()

    def partial_book(self, book_id: int, reason: str) -> None:
        """Mark a book partially complete.

        Args:
            book_id: KCAC item id.
            reason: Partial-run reason.
        """
        entry = self._entry(book_id)
        entry["status"] = "partial"
        entry["reason"] = reason
        entry["updated_at"] = utc_now()
        self._save()

    def fail_book(self, book_id: int, reason: str) -> None:
        """Mark a book failed.

        Args:
            book_id: KCAC item id.
            reason: Failure reason.
        """
        entry = self._entry(book_id)
        entry["status"] = "failed"
        entry["reason"] = reason
        entry["updated_at"] = utc_now()
        self._save()

    def _entry(self, book_id: int) -> dict[str, Any]:
        entry = self._data.setdefault(
            str(book_id),
            {"status": "in_progress", "completed_pages": [], "failed_pages": []},
        )
        return entry

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)


def utc_now() -> str:
    """Return the current UTC timestamp for logs and progress."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def setup_logging(output_dir: Path) -> None:
    """Configure console, debug-file, and error-file logging.

    Args:
        output_dir: Dataset output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("kcac")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    debug_file = logging.FileHandler(output_dir / "scrape.log", encoding="utf-8")
    debug_file.setLevel(logging.DEBUG)
    debug_file.setFormatter(formatter)
    logger.addHandler(debug_file)

    error_file = logging.FileHandler(output_dir / "errors.log", encoding="utf-8")
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(formatter)
    logger.addHandler(error_file)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument vector.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Download books from the KCAC digital archive at full resolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", metavar="YAML", help="Optional YAML config file.")
    id_group = parser.add_mutually_exclusive_group()
    id_group.add_argument("--book-ids", metavar="IDS", help="Comma-separated book IDs.")
    id_group.add_argument(
        "--book-ids-file",
        metavar="FILE",
        help="Text file with one KCAC book id per line.",
    )
    parser.add_argument("--output", default=None, metavar="DIR", help="Dataset output directory.")
    parser.add_argument(
        "--tile-delay",
        type=float,
        default=None,
        metavar="SEC",
        help="Seconds to sleep between tile requests.",
    )
    parser.add_argument(
        "--book-delay",
        type=float,
        default=None,
        metavar="SEC",
        help="Seconds to sleep between books.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        metavar="N",
        help="Retry attempts for connection errors.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        metavar="N",
        help="Maximum concurrent books. Values other than 1 are forced to 1.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        metavar="Q",
        help="JPEG quality for stitched page images.",
    )
    parser.add_argument(
        "--download-thumbs",
        action="store_true",
        default=None,
        help="Also download page thumbnails into thumbs/.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="KCAC archive base URL.",
    )
    return parser.parse_args(argv)


def load_yaml_config(path: str | None) -> dict[str, Any]:
    """Load an optional YAML config file.

    Args:
        path: YAML path or None.

    Returns:
        Parsed config mapping.
    """
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise SetupError("pyyaml is required to use --config.") from exc

    config_path = Path(path)
    if not config_path.exists():
        raise SetupError(f"Config file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SetupError(f"Config file must contain a YAML mapping: {config_path}")
    return data


def config_value(
    args: argparse.Namespace,
    yaml_data: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    """Read one setting with CLI-over-YAML precedence.

    Args:
        args: Parsed CLI args.
        yaml_data: Config file mapping.
        key: Setting key.
        default: Default value.

    Returns:
        Resolved setting value.
    """
    cli_value = getattr(args, key)
    if cli_value is not None:
        return cli_value
    return yaml_data.get(key, default)


def parse_book_ids_text(value: str) -> list[int]:
    """Parse comma/newline-separated book ids.

    Args:
        value: Raw id text.

    Returns:
        Parsed integer ids.
    """
    normalized = value.replace("\n", ",")
    ids: list[int] = []
    for part in normalized.split(","):
        stripped = part.strip()
        if not stripped or stripped.startswith("#"):
            continue
        ids.append(int(stripped))
    return ids


def load_book_ids(args: argparse.Namespace, yaml_data: dict[str, Any]) -> list[int]:
    """Load book ids from CLI or YAML.

    Args:
        args: Parsed CLI args.
        yaml_data: Config file mapping.

    Returns:
        Book id list.

    Raises:
        SetupError: If no valid ids are provided.
    """
    if args.book_ids:
        source = args.book_ids
        try:
            ids = parse_book_ids_text(source)
        except ValueError as exc:
            raise SetupError(f"Invalid --book-ids value: {source}") from exc
        if not ids:
            raise SetupError("--book-ids did not contain any ids.")
        return ids

    book_ids_file = args.book_ids_file or yaml_data.get("book_ids_file")
    if book_ids_file:
        path = Path(str(book_ids_file))
        if not path.exists():
            raise SetupError(f"Book IDs file not found: {path}")
        ids: list[int] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                ids.append(int(stripped))
            except ValueError as exc:
                raise SetupError(f"Invalid book id in {path}: {stripped}") from exc
        if not ids:
            raise SetupError(f"No book IDs found in {path}")
        return ids

    yaml_ids = yaml_data.get("book_ids")
    if isinstance(yaml_ids, list):
        try:
            ids = [int(value) for value in yaml_ids]
        except (TypeError, ValueError) as exc:
            raise SetupError("Config book_ids must be integers.") from exc
        if not ids:
            raise SetupError("Config book_ids is empty.")
        return ids
    if isinstance(yaml_ids, str):
        try:
            ids = parse_book_ids_text(yaml_ids)
        except ValueError as exc:
            raise SetupError("Config book_ids string contains a non-integer.") from exc
        if not ids:
            raise SetupError("Config book_ids is empty.")
        return ids

    raise SetupError("Provide --book-ids, --book-ids-file, or book_ids in config YAML.")


def build_config(args: argparse.Namespace, yaml_data: dict[str, Any], book_ids: list[int]) -> Config:
    """Build and validate runtime configuration.

    Args:
        args: Parsed CLI args.
        yaml_data: Config file mapping.
        book_ids: Book ids to fetch.

    Returns:
        Runtime configuration.
    """
    output_dir = Path(str(config_value(args, yaml_data, "output", "./dataset")))
    tile_delay = float(config_value(args, yaml_data, "tile_delay", 1.5))
    book_delay = float(config_value(args, yaml_data, "book_delay", 5.0))
    max_retries = int(config_value(args, yaml_data, "max_retries", 3))
    max_concurrent = int(config_value(args, yaml_data, "max_concurrent", 1))
    jpeg_quality = int(config_value(args, yaml_data, "jpeg_quality", 95))
    download_thumbs = bool(config_value(args, yaml_data, "download_thumbs", False))
    base_url = str(config_value(args, yaml_data, "base_url", "https://archive.kcac.org"))

    if tile_delay < 0 or book_delay < 0:
        raise SetupError("Delays must be non-negative.")
    if max_retries < 0:
        raise SetupError("--max-retries must be non-negative.")
    if not 1 <= jpeg_quality <= 95:
        raise SetupError("--jpeg-quality must be between 1 and 95.")
    if max_concurrent != 1:
        print("Warning: --max-concurrent must be 1. Forcing to 1.", file=sys.stderr)
        max_concurrent = 1

    return Config(
        output_dir=output_dir,
        book_ids=book_ids,
        tile_delay=tile_delay,
        book_delay=book_delay,
        max_retries=max_retries,
        max_concurrent=max_concurrent,
        jpeg_quality=jpeg_quality,
        download_thumbs=download_thumbs,
        base_url=base_url.rstrip("/"),
    )


def write_metadata(
    book_dir: Path,
    raw_meta: dict[str, Any],
    book_spec: BookSpec,
    started_at: datetime,
    finished_at: datetime | None,
) -> None:
    """Write cleaned metadata for a book.

    Args:
        book_dir: Book output directory.
        raw_meta: Raw KCAC metadata.
        book_spec: Parsed book spec.
        started_at: Scrape start timestamp.
        finished_at: Scrape finish timestamp.
    """
    metadata = parse_meta(raw_meta, book_spec, started_at=started_at, finished_at=finished_at)
    (book_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def valid_page_paths(book_spec: BookSpec, pages_dir: Path) -> list[Path]:
    """Return valid page files in book order.

    Args:
        book_spec: Parsed book spec.
        pages_dir: Pages directory.

    Returns:
        Existing valid page paths.
    """
    paths: list[Path] = []
    for page in book_spec.pages:
        path = pages_dir / f"page_{page.label:04d}.jpg"
        if page_exists_and_valid(path, page.width, page.height):
            paths.append(path)
    return paths


def download_thumbnail_best_effort(
    session: requests.Session,
    cfg: Config,
    state: RequestState,
    page_id: int,
    thumb_path: Path,
) -> None:
    """Download a thumbnail without failing the page if it errors.

    Args:
        session: Shared HTTP session.
        cfg: Runtime configuration.
        state: Mutable per-run request state.
        page_id: KCAC page id.
        thumb_path: Output thumbnail path.
    """
    try:
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        thumb_path.write_bytes(fetch_thumbnail(session, cfg, state, page_id))
    except (AuthenticationRequired, RateLimitAbort):
        raise
    except Exception as exc:
        log.debug("Thumbnail fetch failed for page_id=%d: %s", page_id, exc)


def process_book(
    book_id: int,
    cfg: Config,
    session: requests.Session,
    state: RequestState,
    progress: ProgressTracker,
) -> BookResult:
    """Download one KCAC book.

    Args:
        book_id: KCAC item id.
        cfg: Runtime configuration.
        session: Shared HTTP session.
        state: Mutable per-run request state.
        progress: Progress tracker.

    Returns:
        Book run summary.
    """
    book_dir = cfg.output_dir / str(book_id)
    pages_dir = book_dir / "pages"
    thumbs_dir = book_dir / "thumbs"
    pages_dir.mkdir(parents=True, exist_ok=True)
    if cfg.download_thumbs:
        thumbs_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(tz=timezone.utc)
    raw_meta = fetch_meta(session, cfg, state, book_id)
    pages_data = fetch_pages(session, cfg, state, book_id)
    book_spec = parse_book_spec(book_id, pages_data)
    progress.start_book(book_id, book_spec.total_pages)
    write_metadata(book_dir, raw_meta, book_spec, started_at, None)

    pdf_path = book_dir / f"{book_id}.pdf"
    if pdf_exists_and_valid(pdf_path, book_spec.total_pages):
        progress.complete_book(book_id)
        existing_pages = valid_page_paths(book_spec, pages_dir)
        log.info("[book=%d] PDF already complete; skipping.", book_id)
        return BookResult(
            book_id=book_id,
            expected_pages=book_spec.total_pages,
            downloaded_pages=len(existing_pages) or book_spec.total_pages,
            size_mb=total_size_mb(existing_pages),
            status="complete",
        )

    consecutive_failures = 0
    failed_pages: list[int] = []
    suspect_pages: list[int] = []
    aborted_reason: str | None = None

    page_bar = tqdm(book_spec.pages, desc=f"book={book_id}", unit="page", leave=False)
    for page in page_bar:
        page_bar.set_postfix(page=f"{page.label}/{book_spec.total_pages}")
        page_path = pages_dir / f"page_{page.label:04d}.jpg"

        if page_exists_and_valid(page_path, page.width, page.height):
            progress.mark_page_done(book_id, page.label)
            consecutive_failures = 0
            log.info(
                "[book=%d page=%d/%d (id=%d)] SKIP existing %dx%d px",
                book_id,
                page.label,
                book_spec.total_pages,
                page.id,
                page.width,
                page.height,
            )
            continue

        if cfg.download_thumbs:
            thumb_path = thumbs_dir / f"page_{page.label:04d}.jpg"
            download_thumbnail_best_effort(session, cfg, state, page.id, thumb_path)

        start = time.monotonic()
        partial_dir = pages_dir / f"page_{page.label:04d}.partial"
        try:
            result = stitch_page(page, session, cfg, state, partial_dir)
        except StitchError as exc:
            failed_pages.append(page.label)
            consecutive_failures += 1
            progress.mark_page_failed(book_id, page.label, str(exc))
            log.error(
                "[book=%d page=%d/%d (id=%d)] FAILED %s",
                book_id,
                page.label,
                book_spec.total_pages,
                page.id,
                exc,
            )
            if consecutive_failures >= 5:
                aborted_reason = "5 consecutive page failures"
                log.error("[book=%d] Aborting book: %s", book_id, aborted_reason)
                break
            continue

        size_bytes = save_page_jpeg(result.image, page_path, cfg.jpeg_quality)
        elapsed = time.monotonic() - start
        valid = verify_page(page_path, page.width, page.height)
        if size_bytes < 100 * 1024:
            valid = False
            log.warning("[book=%d page=%d] JPEG is under 100 KB.", book_id, page.label)

        if not valid:
            suspect_pages.append(page.label)
        progress.mark_page_done(book_id, page.label, suspect=not valid)
        consecutive_failures = 0

        log.info(
            "[book=%d page=%d/%d (id=%d)] OK %d tiles, %dx%d px, %.1f MB, %.1fs",
            book_id,
            page.label,
            book_spec.total_pages,
            page.id,
            result.tile_count,
            page.width,
            page.height,
            size_bytes / 1_048_576,
            elapsed,
        )

    finished_at = datetime.now(tz=timezone.utc)
    write_metadata(book_dir, raw_meta, book_spec, started_at, finished_at)

    page_paths = valid_page_paths(book_spec, pages_dir)
    downloaded_pages = len(page_paths)
    size_mb = total_size_mb(page_paths)
    if not page_paths:
        reason = aborted_reason or "No valid page JPEGs were downloaded."
        progress.fail_book(book_id, reason)
        return BookResult(book_id, book_spec.total_pages, 0, 0.0, "failed")

    pdf_ok = False
    try:
        assemble_pdf(page_paths, pdf_path)
        pdf_ok = verify_pdf(pdf_path, book_spec.total_pages)
    except Exception as exc:
        log.error("[book=%d] PDF assembly failed: %s", book_id, exc)

    dpi = estimate_dpi(page_paths)
    if dpi is not None:
        log.info("[book=%d] Average DPI estimate: %.0f", book_id, dpi)

    missing_pages = [
        page.label
        for page in book_spec.pages
        if not page_exists_and_valid(pages_dir / f"page_{page.label:04d}.jpg", page.width, page.height)
    ]
    if aborted_reason or failed_pages or suspect_pages or missing_pages or not pdf_ok:
        reason_parts = []
        if aborted_reason:
            reason_parts.append(aborted_reason)
        if failed_pages:
            reason_parts.append(f"failed pages: {failed_pages}")
        if suspect_pages:
            reason_parts.append(f"suspect pages: {suspect_pages}")
        if missing_pages:
            reason_parts.append(f"missing pages: {missing_pages}")
        if not pdf_ok:
            reason_parts.append("PDF page count mismatch")
        progress.partial_book(book_id, "; ".join(reason_parts))
        return BookResult(book_id, book_spec.total_pages, downloaded_pages, size_mb, "partial")

    progress.complete_book(book_id)
    log.info(
        "[BOOK %d] %d pages, %.1f MB, %s",
        book_id,
        downloaded_pages,
        size_mb,
        resolution_range(book_spec),
    )
    return BookResult(book_id, book_spec.total_pages, downloaded_pages, size_mb, "complete")


def total_size_mb(paths: list[Path]) -> float:
    """Compute total file size in MiB.

    Args:
        paths: File paths.

    Returns:
        Total size in MiB.
    """
    return sum(path.stat().st_size for path in paths if path.exists()) / 1_048_576


def resolution_range(book_spec: BookSpec) -> str:
    """Format min and max page resolution for a book.

    Args:
        book_spec: Parsed book spec.

    Returns:
        Resolution range string.
    """
    if not book_spec.pages:
        return "no pages"
    dims = [(page.width, page.height) for page in book_spec.pages]
    min_dim = min(dims, key=lambda item: item[0] * item[1])
    max_dim = max(dims, key=lambda item: item[0] * item[1])
    return f"{min_dim[0]}x{min_dim[1]} to {max_dim[0]}x{max_dim[1]}"


def print_summary(results: list[BookResult]) -> None:
    """Print the final run summary table.

    Args:
        results: Per-book run results.
    """
    if not results:
        print("No books were processed.")
        return

    rows = [
        (
            str(result.book_id),
            f"{result.downloaded_pages}/{result.expected_pages}",
            f"{result.size_mb:.1f}",
            result.status,
        )
        for result in results
    ]
    headers = ("book_id", "pages", "size_mb", "status")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    border = "+".join("-" * (width + 2) for width in widths)
    print()
    print(f"+{border}+")
    print(
        "| "
        + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
        + " |"
    )
    print(f"+{border}+")
    for row in rows:
        print(
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
        )
    print(f"+{border}+")


def main(argv: list[str] | None = None) -> int:
    """Run the KCAC fetcher CLI.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit code.
    """
    try:
        args = parse_args(argv)
        yaml_data = load_yaml_config(args.config)
        book_ids = load_book_ids(args, yaml_data)
        cfg = build_config(args, yaml_data, book_ids)
        setup_logging(cfg.output_dir)
    except SetupError as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        return 3

    log.info("KCAC fetch starting: %d book(s), output=%s", len(cfg.book_ids), cfg.output_dir)
    log.info(
        "Politeness: tile_delay=%.1fs, book_delay=%.1fs, max_concurrent=%d",
        cfg.tile_delay,
        cfg.book_delay,
        cfg.max_concurrent,
    )

    session = build_session(cfg)
    state = RequestState()
    progress = ProgressTracker(cfg.output_dir / "progress.json")
    results: list[BookResult] = []

    try:
        check_robots_txt(session, cfg)
    except SetupError as exc:
        log.error("%s", exc)
        return 3

    book_bar = tqdm(cfg.book_ids, desc="books", unit="book")
    for index, book_id in enumerate(book_bar):
        book_bar.set_postfix(book=book_id)
        log.info("--- [%d/%d] book=%d ---", index + 1, len(cfg.book_ids), book_id)
        try:
            result = process_book(book_id, cfg, session, state, progress)
        except AuthenticationRequired as exc:
            log.error("%s", exc)
            return 3
        except RateLimitAbort as exc:
            log.error("%s", exc)
            progress.partial_book(book_id, str(exc))
            results.append(BookResult(book_id, 0, 0, 0.0, "failed"))
            break
        except (PermanentFetchError, requests.RequestException) as exc:
            log.error("[book=%d] Failed: %s", book_id, exc)
            progress.fail_book(book_id, str(exc))
            results.append(BookResult(book_id, 0, 0, 0.0, "failed"))
        except Exception as exc:
            log.exception("[book=%d] Unhandled exception: %s", book_id, exc)
            progress.fail_book(book_id, f"unhandled: {exc}")
            results.append(BookResult(book_id, 0, 0, 0.0, "failed"))
        else:
            results.append(result)

        if index < len(cfg.book_ids) - 1:
            log.info("Sleeping %.1fs before next book.", cfg.book_delay)
            time.sleep(cfg.book_delay)

    print_summary(results)

    if not results:
        return 2
    if all(result.status == "failed" for result in results):
        return 2
    if any(result.status != "complete" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
