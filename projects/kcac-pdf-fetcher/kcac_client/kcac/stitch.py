from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image
import requests

from .api import AuthenticationRequired, RateLimitAbort, fetch_tile
from .config import Config, PageSpec, RequestState

log = logging.getLogger(__name__)


class StitchError(Exception):
    """Raised when a page cannot be assembled from its tiles."""


@dataclass(frozen=True)
class StitchResult:
    """Result of a successful page stitch."""

    image: Image.Image
    tile_count: int


def stitch_page(
    page: PageSpec,
    session: requests.Session,
    cfg: Config,
    state: RequestState,
    partial_dir: Path,
) -> StitchResult:
    """Download and stitch one full-resolution page.

    Args:
        page: Page specification.
        session: Shared HTTP session.
        cfg: Runtime configuration.
        state: Mutable per-run request state.
        partial_dir: Directory used to preserve captured tiles on failure.

    Returns:
        Stitched page image and downloaded tile count.

    Raises:
        StitchError: If any tile permanently fails or cannot be decoded.
    """
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True, exist_ok=True)

    canvas = Image.new("RGB", (page.width, page.height), color="white")
    tile_count = 0

    try:
        for row in range(page.rows):
            for col in range(page.cols):
                url = page.tile_url(row, col)
                log.debug(
                    "GET tile page_id=%d level=%d row=%d col=%d url=%s",
                    page.id,
                    page.max_level,
                    row,
                    col,
                    url,
                )
                try:
                    tile_bytes = fetch_tile(session, cfg, state, url)
                    _save_partial_tile(partial_dir, row, col, tile_bytes)
                    tile_img = decode_tile(tile_bytes, url)
                except (AuthenticationRequired, RateLimitAbort):
                    raise
                except Exception as exc:
                    log.error(
                        "Tile failed for page_id=%d row=%d col=%d: %s",
                        page.id,
                        row,
                        col,
                        exc,
                    )
                    time.sleep(cfg.tile_delay)
                    raise StitchError(
                        f"Failed tile row={row} col={col} for page_id={page.id}: {exc}"
                    ) from exc

                canvas.paste(tile_img, (col * page.tile.width, row * page.tile.height))
                tile_count += 1
                time.sleep(cfg.tile_delay)
    except StitchError:
        log.error("Preserved partial tile set in %s", partial_dir)
        raise

    shutil.rmtree(partial_dir)
    return StitchResult(image=canvas, tile_count=tile_count)


def decode_tile(tile_bytes: bytes, url: str) -> Image.Image:
    """Decode tile bytes into an RGB Pillow image.

    Args:
        tile_bytes: Raw tile bytes.
        url: Source URL for error messages.

    Returns:
        Loaded RGB image.
    """
    try:
        with Image.open(BytesIO(tile_bytes)) as tile:
            tile.load()
            return tile.convert("RGB")
    except Exception as exc:
        raise StitchError(f"Could not decode tile {url}") from exc


def save_page_jpeg(image: Image.Image, path: Path, quality: int = 95) -> int:
    """Save a stitched page as a high-quality JPEG.

    Args:
        image: Stitched page image.
        path: Output path.
        quality: JPEG quality.

    Returns:
        File size in bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=quality, subsampling=0, optimize=False)
    return path.stat().st_size


def verify_page(path: Path, expected_width: int, expected_height: int) -> bool:
    """Verify that a page JPEG is readable and has the expected dimensions.

    Args:
        path: Page JPEG path.
        expected_width: Expected pixel width.
        expected_height: Expected pixel height.

    Returns:
        True when the image is valid and dimensions match.
    """
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:
        log.warning("Image verification failed for %s: %s", path, exc)
        return False

    if (width, height) != (expected_width, expected_height):
        log.warning(
            "Dimension mismatch for %s: got %dx%d, expected %dx%d",
            path,
            width,
            height,
            expected_width,
            expected_height,
        )
        return False
    return True


def page_exists_and_valid(path: Path, width: int, height: int) -> bool:
    """Return True when an existing page JPEG can be skipped.

    Args:
        path: Page JPEG path.
        width: Expected width.
        height: Expected height.

    Returns:
        True if the file exists, is non-empty, opens cleanly, and matches dimensions.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    return verify_page(path, width, height)


def _save_partial_tile(partial_dir: Path, row: int, col: int, data: bytes) -> None:
    path = partial_dir / f"tile_r{row:04d}_c{col:04d}.jpg"
    path.write_bytes(data)
