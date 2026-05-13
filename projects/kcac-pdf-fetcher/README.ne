# KCAC PDF And OCR Dataset Project

This repository contains tools and generated dataset files for building a
research dataset from the public KCAC Digital Archive at
`https://archive.kcac.org`.

The main project goal is to download publicly visible KCAC book pages at full
resolution, stitch page tiles into complete page images, assemble image-based
PDF files, extract KCAC OCR/Clip text, and keep validation/report files that
make the dataset usable for Kurdish OCR research.

The working client is in `kcac_client/`. The root folder also contains early
discovery/probe utilities used to understand the KCAC and possible IIIF-style
viewer APIs.

## What This Project Does

- Reads KCAC book metadata from `/api/item/{book_id}/meta`.
- Reads KCAC page/tile metadata from `/api/item/{book_id}/pages`.
- Downloads the highest-resolution tiles for each page.
- Stitches tiles into one full-page JPEG per page.
- Builds one PDF per book from the page JPEG files.
- Extracts OCR text from the public KCAC Clip endpoint.
- Saves per-page text, raw OCR JSON, combined TXT, and TSV line files.
- Validates page counts, image integrity, PDF page counts, and missing pages.
- Supports resumable runs, so interrupted downloads can continue.
- Uses conservative request delays, `robots.txt` checks, and an identifying
  User-Agent.

## What This Project Does Not Do

- It does not log in to KCAC.
- It does not bypass authentication, paywalls, or access controls.
- It does not remove watermarks or alter source rights information.
- It does not make OCR text searchable inside the generated PDFs; OCR text is
  stored separately in UTF-8 `.txt`, `.tsv`, and `.json` files.
- It does not include the full Surya annotation pipeline code. Book `409`
  currently contains generated Surya-style annotation artifacts, but those are
  dataset outputs rather than part of the checked-in KCAC fetch client.

## Repository Layout

```text
.
|-- README.ne
|-- iiif_discover.py
|-- iiif_discover_399.json
|-- kcac_probe.py
|-- probe_output/
|-- probe_output_409/
|-- prompts/
`-- kcac_client/
    |-- README.md
    |-- requirements.txt
    |-- config.yaml.example
    |-- kcac_fetch.py
    |-- kcac_text_extract.py
    |-- validate.py
    |-- kcac/
    |   |-- api.py
    |   |-- config.py
    |   |-- metadata.py
    |   |-- pdf.py
    |   |-- stitch.py
    |   `-- text.py
    `-- dataset/
```

Important scripts:

| Path | Purpose |
| --- | --- |
| `kcac_client/kcac_fetch.py` | Main downloader. Fetches metadata, page images, optional thumbnails, and builds PDFs. |
| `kcac_client/kcac_text_extract.py` | OCR/Clip text extractor. Writes per-page text and combined OCR line files. |
| `kcac_client/validate.py` | Dataset validator. Writes a CSV report and returns a nonzero status if problems are found. |
| `kcac_probe.py` | One-off KCAC API schema probe. Writes raw probe JSON under `probe_output/`. |
| `iiif_discover.py` | Browser-based IIIF endpoint discovery helper. Useful for checking whether a viewer exposes IIIF endpoints. |

## Requirements

Required for the KCAC client:

- Python 3.10 or newer
- Network access to `https://archive.kcac.org`
- Enough disk space for page JPEGs and PDFs
- Python packages from `kcac_client/requirements.txt`

The required Python packages are:

```text
requests
Pillow
img2pdf
pypdf
tqdm
pyyaml
beautifulsoup4
```

Optional for `iiif_discover.py`:

```text
playwright
```

After installing Playwright, Chromium must also be installed:

```bash
playwright install chromium
```

Optional for external annotation work:

- CUDA-capable GPU if running heavy document-layout models.
- Surya or another document-layout/text-line detector if regenerating
  `json_of_pages/` annotations.

## Installation

From the repository root:

```bash
cd kcac_client
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
cd kcac_client
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Recommended Workflow

1. Choose one or more KCAC book IDs from URLs like
   `https://archive.kcac.org/zoom/409/view`.
2. Download page images and build PDFs with `kcac_fetch.py`.
3. Extract OCR text with `kcac_text_extract.py`.
4. Validate the dataset with `validate.py`.
5. Review or correct the combined TSV files, such as
   `409_ocr_lines_editing.tsv`.
6. Use the page images, OCR text, and optional layout annotations for OCR or
   document-layout research.

## Download Books

Fetch one book:

```bash
python kcac_fetch.py --book-ids 409 --output ./dataset
```

Fetch several books:

```bash
python kcac_fetch.py ^
  --book-ids 399,402,403,404,405,406,407,408,409,410 ^
  --output ./dataset ^
  --tile-delay 1.5 ^
  --book-delay 5.0 ^
  --max-retries 3 ^
  --max-concurrent 1
```

PowerShell also accepts the same command on one line:

```bash
python kcac_fetch.py --book-ids 409 --output ./dataset --tile-delay 1.5 --book-delay 5.0
```

Fetch from a text file:

```bash
python kcac_fetch.py --book-ids-file books.txt --output ./dataset
```

`books.txt` format:

```text
399
402
409
# comments and blank lines are ignored
```

Use a YAML config:

```bash
python kcac_fetch.py --config config.yaml.example
```

CLI flags override YAML settings.

## Extract OCR Text

Extract OCR text for one book:

```bash
python kcac_text_extract.py --book-ids 409 --output ./dataset
```

Extract only one page:

```bash
python kcac_text_extract.py --book-ids 409 --output ./dataset --start-page 18 --end-page 18
```

Extract OCR text for every numeric book folder already under `dataset/`:

```bash
python kcac_text_extract.py --all-books-in-output --output ./dataset
```

Force re-extraction of existing text:

```bash
python kcac_text_extract.py --book-ids 409 --output ./dataset --force
```

Skip raw JSON output:

```bash
python kcac_text_extract.py --book-ids 409 --output ./dataset --no-raw
```

## Validate The Dataset

Run validation:

```bash
python validate.py --output ./dataset --report validation.csv
```

The validator checks:

- Numeric book directories under the output folder.
- `metadata.json` page count.
- Existing `pages/page_NNNN.jpg` files.
- Missing page labels.
- Image readability and expected dimensions.
- PDF existence and PDF page count.
- Total page-image size and resolution statistics.

The validation report is a CSV with these columns:

```text
book_id,expected_pages,downloaded_pages,missing_pages,min_resolution,
max_resolution,avg_resolution_mp,total_size_mb,pdf_exists,pdf_page_count,
pdf_matches_json,status
```

Validation status values:

| Status | Meaning |
| --- | --- |
| `complete` | Pages, metadata, and PDF match. |
| `partial` | Some data exists, but pages, images, or PDF output need review. |
| `pdf_mismatch` | PDF exists but page count does not match metadata. |
| `failed` | No valid pages were found for that book. |

## Dataset Output Format

Normal fetch/OCR output:

```text
dataset/
|-- progress.json
|-- scrape.log
|-- errors.log
|-- ocr_progress.json
|-- ocr.log
|-- ocr_errors.log
`-- 409/
    |-- metadata.json
    |-- pages/
    |   |-- page_0001.jpg
    |   |-- page_0002.jpg
    |   `-- ...
    |-- thumbs/
    |   `-- page_0001.jpg
    |-- text/
    |   |-- page_0001.txt
    |   |-- page_0002.txt
    |   `-- ...
    |-- text_raw/
    |   |-- page_0001.json
    |   |-- page_0002.json
    |   `-- ...
    |-- 409.pdf
    |-- 409_ocr_lines.txt
    |-- 409_ocr_lines.tsv
    `-- 409_ocr_lines_editing.tsv
```

`thumbs/` is optional and appears only when `--download-thumbs` is used.

Book `409` also currently contains layout/annotation artifacts:

```text
dataset/409/json_of_pages/
|-- annotations/
|   |-- coco_annotations_YYYYMMDD_HHMMSS.json
|   `-- per_image/
|       |-- page_0001.json
|       |-- page_0002.json
|       `-- ...
|-- logs/
|   |-- pipeline.log
|   `-- report_YYYYMMDD_HHMMSS.json
`-- visualizations/
    |-- page_0001_viz.jpg
    |-- page_0002_viz.jpg
    `-- ...
```

## File Format Details

### `metadata.json`

Cleaned metadata for one KCAC book.

Important fields:

| Field | Meaning |
| --- | --- |
| `book_id` | KCAC item ID. |
| `source_url` | KCAC viewer URL for citation and traceability. |
| `title`, `title_plain` | Title from KCAC metadata. |
| `authors` | Author/creator values from KCAC metadata. |
| `publisher` | Publisher value, when available. |
| `place_of_publication` | Publication place, when available. |
| `publication_date` | Normalized date when parseable. |
| `publication_date_raw` | Original KCAC date value. |
| `language` | Language value from KCAC. |
| `category`, `tags` | KCAC category and tags. |
| `source_collection` | KCAC collection/source field. |
| `page_count` | Expected number of pages. |
| `page_resolutions` | List of `[width, height]` for each page. |
| `scrape_started_at` | UTC timestamp for fetch start. |
| `scrape_finished_at` | UTC timestamp for fetch completion. |
| `tile_protocol` | Internal protocol label, currently `kcac-osd-v1`. |
| `license_note` | Citation/reminder text. |

### `pages/page_NNNN.jpg`

Full-page image for one page.

Format rules:

- JPEG image.
- RGB mode.
- One-based page label with four digits, for example `page_0001.jpg`.
- Dimensions should match the matching entry in `metadata.json`.
- Built by stitching KCAC page tiles at the highest available tile level.
- Saved at high JPEG quality, default `95`.

### `{book_id}.pdf`

Image-based PDF assembled from the page JPEG files.

Notes:

- The PDF is built with `img2pdf`.
- Page order follows sorted `page_NNNN.jpg` filenames.
- The PDF should have the same page count as `metadata.json`.
- OCR text is not embedded in the PDF by this client.

### `text/page_NNNN.txt`

UTF-8 OCR text for one page.

Rules:

- One file per page.
- Each non-empty OCR line is written on one line.
- Whitespace is normalized.
- If KCAC returns one large text block without line geometry, it remains one
  line because the Clip endpoint does not expose physical line boxes in that
  response.

### `text_raw/page_NNNN.json`

Raw response from the KCAC Clip endpoint.

Typical structure:

```json
[
  {
    "rect": {
      "x": 0,
      "y": 0,
      "width": 1733,
      "height": 2480
    },
    "pageId": 40118,
    "type": "text",
    "id": "c-409-0001",
    "text": "..."
  }
]
```

### `{book_id}_ocr_lines.txt`

Combined OCR text for the whole book.

- Built from all existing `text/page_NNNN.txt` files.
- Contains all lines in page order.
- UTF-8 encoded.

### `{book_id}_ocr_lines.tsv`

Combined OCR lines in tab-separated format.

Columns:

```text
page	line	text
```

Column meanings:

| Column | Meaning |
| --- | --- |
| `page` | One-based page label. |
| `line` | One-based line number inside that page text file. |
| `text` | OCR text with tabs replaced by spaces. |

### `{book_id}_ocr_lines_editing.tsv`

Manual editing/review version of the OCR line TSV.

Expected format is the same as `{book_id}_ocr_lines.tsv`:

```text
page	line	text
```

Recommended use:

- Keep `page` and `line` stable so edited lines can be matched back to the
  original OCR output.
- Edit only the `text` column when correcting OCR.
- Save as UTF-8.
- Do not use commas as separators; this file is tab-separated.

### `json_of_pages/annotations/per_image/page_NNNN.json`

Per-page document-layout/text-line annotation file.

Observed fields:

| Field | Meaning |
| --- | --- |
| `file` | Matching page image filename. |
| `width`, `height` | Image dimensions. |
| `bbox_format` | Bounding-box format, currently `coco_xywh`. |
| `text_lines` | Text-line boxes with `bbox` and `confidence`. |
| `layout` | Layout regions with `bbox`, `label`, and `score`. |

Bounding boxes use COCO `xywh`:

```text
[x, y, width, height]
```

Coordinates are pixel coordinates relative to the top-left corner of the page
image.

### `json_of_pages/annotations/coco_annotations_*.json`

COCO-style annotation export for model training or bootstrapping.

Expected COCO concepts:

- `images`
- `annotations`
- `categories`
- image dimensions and file names
- bounding boxes in `xywh` format

### `json_of_pages/logs/report_*.json`

Quality/evaluation summary for generated page annotations.

Observed summary fields include:

- `total`
- `good_count`
- `medium_count`
- `poor_count`
- `good_rate`
- `usable_rate`
- `avg_confidence`
- `avg_coverage_pct`
- `total_lines`
- `decision`
- `verdict`

For book `409`, the saved report currently shows 121 total pages, 120 good
pages, 1 medium page, and a `PROCEED` decision.

## Progress And Logs

Downloader files:

| File | Purpose |
| --- | --- |
| `dataset/progress.json` | Per-book and per-page fetch progress. |
| `dataset/scrape.log` | Detailed downloader log. |
| `dataset/errors.log` | Downloader errors only. |

OCR files:

| File | Purpose |
| --- | --- |
| `dataset/ocr_progress.json` | Per-book and per-page OCR progress. |
| `dataset/ocr.log` | Detailed OCR extraction log. |
| `dataset/ocr_errors.log` | OCR extraction errors only. |

Common status values:

- `in_progress`
- `complete`
- `partial`
- `failed`

## Resume Behavior

The downloader can be rerun with the same command.

It skips a page when:

- `pages/page_NNNN.jpg` already exists.
- The JPEG opens cleanly with Pillow.
- The JPEG dimensions match KCAC page metadata.

It skips a whole book when:

- `{book_id}.pdf` exists.
- The PDF page count matches KCAC metadata.

The OCR extractor skips a page when:

- `text/page_NNNN.txt` already exists and is readable as UTF-8.

Use `--force` with `kcac_text_extract.py` to rebuild OCR text files.

## Politeness And Safety Rules

Default behavior is intentionally conservative:

- `tile_delay`: `1.5` seconds between tile requests.
- `page_delay`: `2.0` seconds between OCR/Clip page requests.
- `book_delay`: `5.0` seconds between books.
- `max_concurrent`: forced to `1`.
- `robots.txt` is checked before the main KCAC client runs.
- User-Agent is:
  `Kurdish-OCR-Research/1.0 (PhD dissertation; academic use)`.
- HTTP `401` or `403` stops the run.
- HTTP `429` or `503` uses backoff of `10`, `30`, `90`, and `270` seconds.
- Five consecutive `429` or `503` responses stop the run gracefully.

Increase delays if KCAC returns rate-limit responses.

## Configuration

Example config path:

```text
kcac_client/config.yaml.example
```

Important config keys:

| Key | Meaning | Default/example |
| --- | --- | --- |
| `base_url` | KCAC archive base URL. | `https://archive.kcac.org` |
| `output` | Dataset output folder. | `./dataset` |
| `book_ids` | List of KCAC item IDs. | `[399, 2367, ...]` |
| `book_ids_file` | Optional file containing one ID per line. | `books.txt` |
| `tile_delay` | Delay between page tile requests. | `1.5` |
| `page_delay` | Delay between OCR requests. | `2.0` |
| `book_delay` | Delay between books. | `5.0` |
| `max_retries` | Connection retry attempts. | `3` |
| `jpeg_quality` | Stitched JPEG quality. | `95` |
| `download_thumbs` | Whether to fetch thumbnails. | `false` |
| `ocr_margin` | Pixel inset from page edges for OCR extraction. | `0` |

## Exit Codes

`kcac_fetch.py`:

| Code | Meaning |
| --- | --- |
| `0` | All requested books completed. |
| `1` | At least one requested book was partial. |
| `2` | No books processed, or all failed. |
| `3` | Setup/auth/robots/config problem. |
| `130` | Interrupted by user. |

`kcac_text_extract.py`:

| Code | Meaning |
| --- | --- |
| `0` | All requested OCR extraction completed. |
| `1` | At least one requested OCR extraction was partial. |
| `2` | No OCR results, or all failed. |
| `3` | Setup/auth/robots/config problem. |

`validate.py`:

| Code | Meaning |
| --- | --- |
| `0` | All validated books are complete. |
| `1` | At least one book has a validation issue. |
| `3` | Dataset directory or input setup problem. |

## IIIF Discovery Utility

`iiif_discover.py` is a root-level helper for checking whether a viewer exposes
IIIF Presentation or Image API endpoints.

Example:

```bash
python iiif_discover.py --viewer-url https://archive.kcac.org/zoom/399/view
```

It writes a JSON network log such as:

```text
iiif_discover_399.json
```

The log contains:

- `viewer_url`
- `summary.manifest_urls`
- `summary.info_json_urls`
- `summary.image_api_urls`
- `summary.image_service_urls`
- `network` entries with URL, method, resource type, status, content type, and
  matched signals

This utility only reports IIIF evidence. It does not fetch non-IIIF books.

## KCAC API Probe Utility

`kcac_probe.py` is a one-off schema probe.

Example:

```bash
python kcac_probe.py 409
```

It writes raw inspection files under `probe_output/`:

```text
probe_output/
|-- item_409_meta.json
|-- item_409_pages.json
`-- tile_levels_map.json
```

Use this only when checking or debugging KCAC API structure.

## Current Dataset Notes

The current workspace contains numeric dataset folders including:

```text
399, 402, 403, 404, 405, 406, 407, 408, 409, 410
```

Book `409` is the most complete visible example in this workspace. It includes:

- `metadata.json`
- `pages/page_0001.jpg` through `pages/page_0121.jpg`
- `409.pdf`
- per-page OCR text under `text/`
- raw OCR JSON under `text_raw/`
- combined OCR files:
  - `409_ocr_lines.txt`
  - `409_ocr_lines.tsv`
  - `409_ocr_lines_editing.tsv`
- layout/annotation outputs under `json_of_pages/`

## Troubleshooting

Missing Python package:

```bash
pip install -r requirements.txt
```

HTTP `429` or `503`:

- Increase `--tile-delay`.
- Increase `--page-delay` for OCR extraction.
- Increase `--book-delay`.
- Rerun later.

Interrupted download:

- Rerun the same `kcac_fetch.py` command.
- Existing valid page JPEGs are skipped.

Interrupted OCR extraction:

- Rerun the same `kcac_text_extract.py` command.
- Existing UTF-8 text files are skipped unless `--force` is used.

PDF page count mismatch:

- Run `python validate.py --output ./dataset --report validation.csv`.
- Check `missing_pages` and `status`.
- Delete only bad/missing generated files if needed.
- Rerun the fetch command.

Arabic-script/Kurdish text looks broken in a terminal:

- Files are written as UTF-8.
- Open `.txt` or `.tsv` files in VS Code or another UTF-8-aware editor.
- Make sure the editor font supports Arabic-script text.

Endpoint returns `401` or `403`:

- Stop the run.
- Do not attempt bypass logic.
- Contact KCAC if access is needed.

## Citation And Use

This project is intended for academic research using publicly accessible KCAC
archive pages. Cite KCAC as the source when using downloaded images, PDFs, OCR
text, or derived annotations.

Suggested citation text:

```text
Kurdistan Center for Arts and Culture Digital Archive.
https://archive.kcac.org
```

Do not redistribute downloaded images or PDFs commercially.

