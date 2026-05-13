"""
KCAC API Probe — run this once to capture the API schema.

Usage:
    python kcac_probe.py 399

It saves three files to ./probe_output/ and prints a quick summary:
    - item_399_meta.json
    - item_399_pages.json
    - tile_levels_map.json   (tries levels 0..12 on the first page)

Paste the printed summary back to your AI assistant so it can write
the fetcher with full knowledge of the API.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import requests

BASE = "https://archive.kcac.org"
HEADERS = {
    "User-Agent": "Kurdish-OCR-Research/1.0 (PhD dissertation; academic use)",
    "Accept": "application/json, image/jpeg, */*",
}
OUT = Path("./probe_output")
OUT.mkdir(exist_ok=True)


def get_json(path: str) -> dict | list:
    r = requests.get(BASE + path, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def head_image(path: str) -> tuple[int, int]:
    """Returns (status_code, content_length). 200 = tile exists."""
    r = requests.head(BASE + path, headers=HEADERS, timeout=30, allow_redirects=True)
    cl = int(r.headers.get("Content-Length") or 0)
    return r.status_code, cl


def get_image_size(path: str) -> tuple[int, int] | None:
    """Download tile and return (width, height) using Pillow."""
    from io import BytesIO
    try:
        from PIL import Image
    except ImportError:
        return None
    r = requests.get(BASE + path, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None
    img = Image.open(BytesIO(r.content))
    return img.size


def main(book_id: str) -> None:
    print(f"[1/4] GET /api/item/{book_id}/meta")
    meta = get_json(f"/api/item/{book_id}/meta")
    (OUT / f"item_{book_id}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"     keys: {list(meta.keys()) if isinstance(meta, dict) else 'list'}")

    time.sleep(1)
    print(f"[2/4] GET /api/item/{book_id}/pages")
    pages = get_json(f"/api/item/{book_id}/pages")
    (OUT / f"item_{book_id}_pages.json").write_text(
        json.dumps(pages, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if isinstance(pages, list):
        print(f"     total pages: {len(pages)}")
        if pages:
            print(f"     first page keys: {list(pages[0].keys()) if isinstance(pages[0], dict) else 'scalar'}")
            print(f"     first page sample: {json.dumps(pages[0], ensure_ascii=False)[:300]}")
    elif isinstance(pages, dict):
        print(f"     dict keys: {list(pages.keys())}")
        # Sometimes pages are wrapped, e.g. {"data": [...], "total": N}
        for k in ("data", "pages", "items", "results"):
            if k in pages and isinstance(pages[k], list):
                print(f"     wrapped under '{k}', count = {len(pages[k])}")

    # Find first page id from various possible shapes
    first_page_id = None
    if isinstance(pages, list) and pages and isinstance(pages[0], dict):
        for k in ("id", "page_id", "_id", "pageId"):
            if k in pages[0]:
                first_page_id = pages[0][k]
                break
    elif isinstance(pages, dict):
        for k in ("data", "pages", "items"):
            if k in pages and isinstance(pages[k], list) and pages[k]:
                p = pages[k][0]
                if isinstance(p, dict):
                    for kk in ("id", "page_id", "_id", "pageId"):
                        if kk in p:
                            first_page_id = p[kk]
                            break
                break

    if first_page_id is None:
        print("     ⚠ could not auto-detect first page id; please inspect the JSON file")
        return

    print(f"     first page id: {first_page_id}")

    time.sleep(1)
    print(f"[3/4] HEAD tiles at levels 0..14 for page {first_page_id} (tile 0,0)")
    levels = {}
    for lvl in range(15):
        path = f"/api/page/{first_page_id}/tile/{lvl}/0/0"
        try:
            status, length = head_image(path)
        except Exception as e:
            status, length = -1, 0
        levels[lvl] = {"status": status, "content_length": length}
        if status == 200:
            print(f"     level {lvl:2d}: 200 OK  ({length} bytes)")
        else:
            print(f"     level {lvl:2d}: {status}")
        time.sleep(0.5)
    (OUT / "tile_levels_map.json").write_text(json.dumps(levels, indent=2), encoding="utf-8")

    # Find the highest working level
    max_level = max((lvl for lvl, v in levels.items() if v["status"] == 200), default=-1)
    print(f"     ➜ highest working level: {max_level}")

    if max_level >= 0:
        time.sleep(1)
        print(f"[4/4] Probe tile dimensions and grid at level {max_level}")
        # Get tile pixel size
        sz = get_image_size(f"/api/page/{first_page_id}/tile/{max_level}/0/0")
        if sz:
            print(f"     tile (0,0) at level {max_level}: {sz[0]} x {sz[1]} px")

        # Probe grid extent at max level
        time.sleep(0.5)
        print(f"     probing grid extent at level {max_level}...")
        max_col = -1
        for c in range(20):
            st, _ = head_image(f"/api/page/{first_page_id}/tile/{max_level}/{c}/0")
            if st == 200:
                max_col = c
            else:
                break
            time.sleep(0.3)
        max_row = -1
        for r in range(30):
            st, _ = head_image(f"/api/page/{first_page_id}/tile/{max_level}/0/{r}")
            if st == 200:
                max_row = r
            else:
                break
            time.sleep(0.3)
        print(f"     grid at level {max_level}: cols 0..{max_col}, rows 0..{max_row}")

        if sz and max_col >= 0 and max_row >= 0:
            tw, th = sz
            est_w = tw * (max_col + 1)
            est_h = th * (max_row + 1)
            print(f"     ➜ estimated full image at level {max_level}: ~{est_w} x {est_h} px")

    print(f"\n✓ Probe outputs saved to ./probe_output/")
    print(f"   Now share these files (especially item_{book_id}_pages.json)")
    print(f"   so the agent can write a precise fetcher.")


if __name__ == "__main__":
    book_id = sys.argv[1] if len(sys.argv) > 1 else "399"
    main(book_id)