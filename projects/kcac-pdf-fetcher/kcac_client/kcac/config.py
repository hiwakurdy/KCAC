from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TileSpec:
    """Tile size for one KCAC page pyramid."""

    width: int
    height: int


@dataclass(frozen=True)
class PageSpec:
    """Description of one full-resolution KCAC page."""

    id: int
    width: int
    height: int
    tile: TileSpec
    levels: int
    label: int
    tile_uri_template: str

    @property
    def max_level(self) -> int:
        """Return the highest-resolution tile level."""
        return self.levels - 1

    @property
    def cols(self) -> int:
        """Return the tile column count at maximum resolution."""
        return math.ceil(self.width / self.tile.width)

    @property
    def rows(self) -> int:
        """Return the tile row count at maximum resolution."""
        return math.ceil(self.height / self.tile.height)

    @property
    def expected_tiles(self) -> int:
        """Return the number of tiles required for the full page."""
        return self.cols * self.rows

    def tile_url(self, row: int, col: int) -> str:
        """Build the tile URL for a row and column.

        Args:
            row: Zero-based tile row.
            col: Zero-based tile column.

        Returns:
            Fully qualified tile URL.
        """
        return self.tile_uri_template.format(
            page_id=self.id,
            level=self.max_level,
            row=row,
            column=col,
        )


@dataclass(frozen=True)
class BookSpec:
    """KCAC book page list and expected page count."""

    book_id: int
    total_pages: int
    pages: list[PageSpec]


@dataclass(frozen=True)
class Config:
    """Runtime configuration for a KCAC fetch run."""

    output_dir: Path
    book_ids: list[int] = field(default_factory=list)
    tile_delay: float = 1.5
    book_delay: float = 5.0
    max_retries: int = 3
    max_concurrent: int = 1
    jpeg_quality: int = 95
    download_thumbs: bool = False
    base_url: str = "https://archive.kcac.org"
    user_agent: str = "Kurdish-OCR-Research/1.0 (PhD dissertation; academic use)"


@dataclass
class RequestState:
    """Mutable per-run HTTP state passed explicitly to request helpers."""

    consecutive_rate_limits: int = 0
