#!/usr/bin/env python3
"""Extract KCAC clip/OCR text for every page of selected books."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from kcac.api import (
    AuthenticationRequired,
    PermanentFetchError,
    RateLimitAbort,
    SetupError,
    build_session,
    check_robots_txt,
    fetch_pages,
    parse_book_spec,
)
from kcac.config import BookSpec, Config, RequestState
from kcac.text import extract_clip_text, text_exists, write_text_outputs

log = logging.getLogger("kcac.ocr")


@dataclass(frozen=True)
class OcrResult:
    """Summary of one OCR extraction run."""

    book_id: int
    expected_pages: int
    completed_pages: int
    status: str


class OcrProgress:
    """JSON-backed OCR extraction progress."""

    def __init__(self, path: Path) -> None:
        """Load progress from disk.

        Args:
            path: Progress JSON path.
        """
        self.path = path
        self.data: dict[str, Any] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("Could not parse %s; starting fresh.", path)

    def start_book(self, book_id: int, expected_pages: int) -> None:
        """Mark a book in progress.

        Args:
            book_id: KCAC item id.
            expected_pages: Expected page count.
        """
        entry = self.data.setdefault(str(book_id), {})
        entry["status"] = "in_progress"
        entry["expected_pages"] = expected_pages
        entry.setdefault("completed_pages", [])
        entry.setdefault("failed_pages", {})
        self.save()

    def mark_done(self, book_id: int, page_label: int) -> None:
        """Mark one page completed.

        Args:
            book_id: KCAC item id.
            page_label: Page label.
        """
        entry = self.data.setdefault(str(book_id), {})
        completed = set(entry.setdefault("completed_pages", []))
        completed.add(page_label)
        entry["completed_pages"] = sorted(completed)
        failed = entry.setdefault("failed_pages", {})
        failed.pop(str(page_label), None)
        entry["status"] = "in_progress"
        self.save()

    def mark_failed(self, book_id: int, page_label: int, reason: str) -> None:
        """Mark one page failed.

        Args:
            book_id: KCAC item id.
            page_label: Page label.
            reason: Failure reason.
        """
        entry = self.data.setdefault(str(book_id), {})
        failed = entry.setdefault("failed_pages", {})
        failed[str(page_label)] = reason
        entry["status"] = "in_progress"
        self.save()

    def finish_book(self, book_id: int, status: str) -> None:
        """Set final status for a book.

        Args:
            book_id: KCAC item id.
            status: Final status.
        """
        entry = self.data.setdefault(str(book_id), {})
        entry["status"] = status
        self.save()

    def save(self) -> None:
        """Write progress JSON atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(self.path)


def setup_logging(output_dir: Path) -> None:
    """Configure OCR extraction logging.

    Args:
        output_dir: Dataset output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("kcac")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    debug_file = logging.FileHandler(output_dir / "ocr.log", encoding="utf-8")
    debug_file.setLevel(logging.DEBUG)
    debug_file.setFormatter(formatter)
    logger.addHandler(debug_file)

    error_file = logging.FileHandler(output_dir / "ocr_errors.log", encoding="utf-8")
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(formatter)
    logger.addHandler(error_file)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument vector.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Extract KCAC OCR text via the public Clip zone endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", metavar="YAML", help="Optional YAML config file.")
    ids = parser.add_mutually_exclusive_group()
    ids.add_argument("--book-ids", help="Comma-separated book IDs, e.g. 399,409.")
    ids.add_argument("--book-ids-file", help="Text file with one book id per line.")
    ids.add_argument(
        "--all-books-in-output",
        action="store_true",
        help="Process every numeric book directory already present under --output.",
    )
    parser.add_argument("--output", default=None, help="Dataset output directory.")
    parser.add_argument("--page-delay", type=float, default=None, help="Seconds between OCR requests.")
    parser.add_argument("--book-delay", type=float, default=None, help="Seconds between books.")
    parser.add_argument("--max-retries", type=int, default=None, help="Connection retry attempts.")
    parser.add_argument("--margin", type=int, default=None, help="Inset OCR rectangle from page edges.")
    parser.add_argument("--start-page", type=int, default=None, help="First page label to process.")
    parser.add_argument("--end-page", type=int, default=None, help="Last page label to process.")
    parser.add_argument("--force", action="store_true", help="Re-extract text files that already exist.")
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Do not write raw clip JSON under text_raw/.",
    )
    parser.add_argument("--base-url", default=None, help="KCAC archive base URL.")
    return parser.parse_args(argv)


def load_yaml_config(path: str | None) -> dict[str, Any]:
    """Load an optional YAML config file.

    Args:
        path: YAML path.

    Returns:
        Parsed YAML mapping.
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


def config_value(args: argparse.Namespace, yaml_data: dict[str, Any], key: str, default: Any) -> Any:
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


def load_book_ids(args: argparse.Namespace, yaml_data: dict[str, Any], output_dir: Path) -> list[int]:
    """Load book ids from CLI, file, or output folders.

    Args:
        args: Parsed CLI args.
        yaml_data: Config file mapping.
        output_dir: Dataset output directory.

    Returns:
        Book id list.

    Raises:
        SetupError: If no ids can be found.
    """
    if args.book_ids:
        try:
            ids = [int(part.strip()) for part in args.book_ids.split(",") if part.strip()]
        except ValueError as exc:
            raise SetupError(f"Invalid --book-ids value: {args.book_ids}") from exc
        if not ids:
            raise SetupError("--book-ids did not contain any ids.")
        return ids

    if args.book_ids_file:
        path = Path(args.book_ids_file)
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
            raise SetupError(f"No book ids found in {path}")
        return ids

    yaml_file = yaml_data.get("book_ids_file")
    if yaml_file:
        path = Path(str(yaml_file))
        if not path.exists():
            raise SetupError(f"Book IDs file not found: {path}")
        ids = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                ids.append(int(stripped))
            except ValueError as exc:
                raise SetupError(f"Invalid book id in {path}: {stripped}") from exc
        if not ids:
            raise SetupError(f"No book ids found in {path}")
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
            ids = [int(part.strip()) for part in yaml_ids.replace("\n", ",").split(",") if part.strip()]
        except ValueError as exc:
            raise SetupError("Config book_ids string contains a non-integer.") from exc
        if not ids:
            raise SetupError("Config book_ids is empty.")
        return ids

    if args.all_books_in_output:
        if not output_dir.exists():
            raise SetupError(f"Output directory does not exist: {output_dir}")
        ids = sorted(int(path.name) for path in output_dir.iterdir() if path.is_dir() and path.name.isdigit())
        if not ids:
            raise SetupError(f"No numeric book directories found in {output_dir}")
        return ids

    raise SetupError("Provide --book-ids, --book-ids-file, or --all-books-in-output.")


def build_config(args: argparse.Namespace, yaml_data: dict[str, Any], book_ids: list[int]) -> Config:
    """Build runtime configuration.

    Args:
        args: Parsed CLI args.
        yaml_data: Config file mapping.
        book_ids: Book ids to process.

    Returns:
        Shared KCAC configuration.
    """
    output_dir = Path(str(config_value(args, yaml_data, "output", "./dataset")))
    page_delay = float(config_value(args, yaml_data, "page_delay", 2.0))
    book_delay = float(config_value(args, yaml_data, "book_delay", 5.0))
    max_retries = int(config_value(args, yaml_data, "max_retries", 3))
    base_url = str(config_value(args, yaml_data, "base_url", "https://archive.kcac.org"))

    if page_delay < 0 or book_delay < 0:
        raise SetupError("Delays must be non-negative.")
    if max_retries < 0:
        raise SetupError("--max-retries must be non-negative.")
    return Config(
        output_dir=output_dir,
        book_ids=book_ids,
        tile_delay=page_delay,
        book_delay=book_delay,
        max_retries=max_retries,
        max_concurrent=1,
        base_url=base_url.rstrip("/"),
    )


def process_book(
    cfg: Config,
    state: RequestState,
    progress: OcrProgress,
    session: Any,
    book_id: int,
    margin: int,
    force: bool,
    write_raw: bool,
    start_page: int | None,
    end_page: int | None,
) -> OcrResult:
    """Extract OCR text for one book.

    Args:
        cfg: Runtime configuration.
        state: Mutable request state.
        progress: OCR progress tracker.
        session: Shared requests session.
        book_id: KCAC item id.
        margin: OCR rectangle inset.
        force: Whether to overwrite existing text files.
        write_raw: Whether to write raw JSON responses.
        start_page: Optional first page label.
        end_page: Optional last page label.

    Returns:
        Book OCR summary.
    """
    pages_data = fetch_pages(session, cfg, state, book_id)
    book_spec = parse_book_spec(book_id, pages_data)
    book_dir = cfg.output_dir / str(book_id)
    book_dir.mkdir(parents=True, exist_ok=True)
    progress.start_book(book_id, book_spec.total_pages)
    pages = [
        page
        for page in book_spec.pages
        if (start_page is None or page.label >= start_page)
        and (end_page is None or page.label <= end_page)
    ]
    if not pages:
        raise SetupError(f"No pages selected for book {book_id}.")

    completed = 0
    failed = 0
    page_bar = tqdm(pages, desc=f"ocr={book_id}", unit="page", leave=False)
    for page in page_bar:
        page_bar.set_postfix(page=f"{page.label}/{book_spec.total_pages}")
        text_path = book_dir / "text" / f"page_{page.label:04d}.txt"
        if not force and text_exists(text_path):
            completed += 1
            progress.mark_done(book_id, page.label)
            log.info("[book=%d page=%d/%d id=%d] SKIP text exists", book_id, page.label, book_spec.total_pages, page.id)
            continue

        start = time.monotonic()
        try:
            zones = extract_clip_text(session, cfg, state, book_id, page, margin=margin)
            _, lines = write_text_outputs(book_dir, page, zones, write_raw=write_raw)
        except (AuthenticationRequired, RateLimitAbort):
            raise
        except Exception as exc:
            failed += 1
            progress.mark_failed(book_id, page.label, str(exc))
            log.error("[book=%d page=%d/%d id=%d] OCR FAILED %s", book_id, page.label, book_spec.total_pages, page.id, exc)
            time.sleep(cfg.tile_delay)
            continue

        completed += 1
        progress.mark_done(book_id, page.label)
        log.info(
            "[book=%d page=%d/%d id=%d] OCR OK %d line(s), %.1fs",
            book_id,
            page.label,
            book_spec.total_pages,
            page.id,
            len(lines),
            time.monotonic() - start,
        )
        time.sleep(cfg.tile_delay)

    rebuild_book_line_files(book_dir, book_spec)
    expected_selected = len(pages)
    status = "complete" if failed == 0 and completed == expected_selected else "partial"
    if completed == 0:
        status = "failed"
    progress.finish_book(book_id, status)
    return OcrResult(book_id, expected_selected, completed, status)


def rebuild_book_line_files(book_dir: Path, book_spec: BookSpec) -> None:
    """Build combined per-book text files from per-page outputs.

    Args:
        book_dir: Book output directory.
        book_spec: Parsed book spec.
    """
    plain_lines: list[str] = []
    tsv_lines = ["page\tline\ttext"]
    for page in book_spec.pages:
        text_path = book_dir / "text" / f"page_{page.label:04d}.txt"
        if not text_path.exists():
            continue
        lines = text_path.read_text(encoding="utf-8").splitlines()
        plain_lines.extend(lines)
        for index, line in enumerate(lines, start=1):
            safe_line = line.replace("\t", " ")
            tsv_lines.append(f"{page.label}\t{index}\t{safe_line}")

    book_id = book_dir.name
    (book_dir / f"{book_id}_ocr_lines.txt").write_text(
        "\n".join(plain_lines) + ("\n" if plain_lines else ""),
        encoding="utf-8",
    )
    (book_dir / f"{book_id}_ocr_lines.tsv").write_text(
        "\n".join(tsv_lines) + "\n",
        encoding="utf-8",
    )


def print_summary(results: list[OcrResult]) -> None:
    """Print final OCR summary.

    Args:
        results: Per-book results.
    """
    print()
    print("OCR summary")
    print("book_id\tpages\tstatus")
    for result in results:
        print(f"{result.book_id}\t{result.completed_pages}/{result.expected_pages}\t{result.status}")


def main(argv: list[str] | None = None) -> int:
    """Run the OCR extraction CLI.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit code.
    """
    try:
        args = parse_args(argv)
        yaml_data = load_yaml_config(args.config)
        output_dir = Path(str(config_value(args, yaml_data, "output", "./dataset")))
        book_ids = load_book_ids(args, yaml_data, output_dir)
        cfg = build_config(args, yaml_data, book_ids)
        margin = int(config_value(args, yaml_data, "margin", yaml_data.get("ocr_margin", 0)))
        if args.start_page is not None and args.end_page is not None and args.start_page > args.end_page:
            raise SetupError("--start-page cannot be greater than --end-page.")
        setup_logging(cfg.output_dir)
    except SetupError as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        return 3

    session = build_session(cfg)
    state = RequestState()
    progress = OcrProgress(cfg.output_dir / "ocr_progress.json")
    results: list[OcrResult] = []

    try:
        check_robots_txt(session, cfg)
    except SetupError as exc:
        log.error("%s", exc)
        return 3

    for index, book_id in enumerate(tqdm(cfg.book_ids, desc="books", unit="book")):
        log.info("--- OCR [%d/%d] book=%d ---", index + 1, len(cfg.book_ids), book_id)
        try:
            result = process_book(
                cfg,
                state,
                progress,
                session,
                book_id,
                margin=margin,
                force=args.force,
                write_raw=not args.no_raw,
                start_page=args.start_page,
                end_page=args.end_page,
            )
        except AuthenticationRequired as exc:
            log.error("%s", exc)
            return 3
        except RateLimitAbort as exc:
            log.error("%s", exc)
            results.append(OcrResult(book_id, 0, 0, "failed"))
            break
        except PermanentFetchError as exc:
            log.error("[book=%d] OCR setup failed: %s", book_id, exc)
            progress.finish_book(book_id, "failed")
            results.append(OcrResult(book_id, 0, 0, "failed"))
        except SetupError as exc:
            log.error("[book=%d] OCR setup failed: %s", book_id, exc)
            progress.finish_book(book_id, "failed")
            results.append(OcrResult(book_id, 0, 0, "failed"))
        except Exception as exc:
            log.exception("[book=%d] Unhandled OCR exception: %s", book_id, exc)
            progress.finish_book(book_id, "failed")
            results.append(OcrResult(book_id, 0, 0, "failed"))
        else:
            results.append(result)

        if index < len(cfg.book_ids) - 1:
            log.info("Sleeping %.1fs before next book.", cfg.book_delay)
            time.sleep(cfg.book_delay)

    print_summary(results)
    if not results or all(result.status == "failed" for result in results):
        return 2
    if any(result.status != "complete" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
