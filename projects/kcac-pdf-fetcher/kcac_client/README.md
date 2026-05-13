# KCAC Digital Library Client

Production-oriented Python client for downloading full-resolution books from the KCAC Digital Archive tile API and assembling OCR-ready PDFs.

## Prerequisites

- Python 3.10 or newer
- Network access to `https://archive.kcac.org`
- Enough disk space for stitched page JPEGs and PDFs

## Installation

```bash
cd kcac_client
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Usage

Fetch one book:

```bash
python kcac_fetch.py --book-ids 399 --output ./dataset
```

Fetch several books:

```bash
python kcac_fetch.py \
  --book-ids 399,2367,276,19,392,911 \
  --output ./dataset \
  --tile-delay 1.5 \
  --book-delay 5.0 \
  --max-retries 3 \
  --max-concurrent 1
```

Fetch from a file:

```bash
python kcac_fetch.py --book-ids-file books.txt --output ./dataset
```

`books.txt` can contain one id per line. Blank lines and lines beginning with `#` are ignored.

Resume a partial run:

```bash
python kcac_fetch.py --book-ids 399 --output ./dataset
```

The client skips page JPEGs that already exist, open cleanly with Pillow, and match the KCAC JSON dimensions. It also skips a book when `{book_id}.pdf` already exists and its page count matches KCAC metadata.

Use a YAML config:

```bash
python kcac_fetch.py --config config.yaml.example
```

CLI flags override YAML settings.

Validate a finished dataset:

```bash
python validate.py --output ./dataset --report validation.csv
```

Extract KCAC Clip/OCR text for one book:

```bash
python kcac_text_extract.py --book-ids 409 --output ./dataset
```

Extract only page 18 of book 409:

```bash
python kcac_text_extract.py --book-ids 409 --output ./dataset --start-page 18 --end-page 18
```

Extract OCR text for every book folder already present in the dataset:

```bash
python kcac_text_extract.py --all-books-in-output --output ./dataset
```

The OCR extractor saves one UTF-8 text file per page:

```text
dataset/409/text/page_0018.txt
```

It also writes:

- `dataset/409/text_raw/page_0018.json` for the raw Clip API response
- `dataset/409/409_ocr_lines.txt` with all non-empty OCR lines
- `dataset/409/409_ocr_lines.tsv` with `page`, `line`, and `text` columns
- `dataset/ocr_progress.json`, `dataset/ocr.log`, and `dataset/ocr_errors.log`

## Output

```text
dataset/
|-- progress.json
|-- scrape.log
|-- errors.log
|-- ocr_progress.json
|-- ocr.log
|-- ocr_errors.log
`-- 399/
    |-- metadata.json
    |-- pages/
    |   |-- page_0001.jpg
    |   `-- page_0002.jpg
    |-- text/
    |   `-- page_0001.txt
    |-- text_raw/
    |   `-- page_0001.json
    |-- thumbs/
    |   `-- page_0001.jpg
    |-- 399_ocr_lines.txt
    |-- 399_ocr_lines.tsv
    `-- 399.pdf
```

Thumbnails are optional and are only downloaded when `--download-thumbs` is set.

## Politeness

Defaults are intentionally conservative:

- 1.5 seconds between tile requests
- 5 seconds between books
- Single connection and no parallel tile fetching
- Identifying User-Agent: `Kurdish-OCR-Research/1.0 (PhD dissertation; academic use)`
- `robots.txt` is checked once at startup

If KCAC returns HTTP 429 or 503, the client backs off for `10`, `30`, `90`, and `270` seconds. On the fifth consecutive 429/503, the run stops gracefully.

## OCR Text Extraction Notes

`kcac_text_extract.py` uses the same public Clip tool visible in the KCAC viewer. For each page it sends one full-page text rectangle to:

```text
/api/item/{book_id}/clip/{page_id}/find
```

The saved page text is normalized so each non-empty OCR line is written as one line in the output file. If KCAC returns the text as a single block without physical line breaks, the script preserves it as a single line because the Clip endpoint does not expose line geometry in that response.

## Troubleshooting

I get HTTP 429:
Increase `--tile-delay`, for example `--tile-delay 3.0`, and consider increasing `--book-delay`.

Run was interrupted:
Rerun the same command. Existing valid pages are skipped and the run continues from the missing pages.

OCR extraction was interrupted:
Rerun `kcac_text_extract.py` with the same arguments. Existing `text/page_NNNN.txt` files are skipped unless `--force` is used.

Arabic-script text looks broken in PowerShell:
The files are written as UTF-8. Some Windows consoles display UTF-8 Arabic/Kurdish text incorrectly unless the terminal encoding/font is configured for it. Open the `.txt` files in VS Code or another UTF-8-aware editor.

PDF page count does not match:
Run `validate.py`, inspect `missing_pages`, delete the bad or missing page JPEGs, then rerun the fetch command.

An endpoint returns 401 or 403:
Stop and contact KCAC. The client does not attempt authentication or bypass logic.

## Ethics And Citation

This client is for academic research on a publicly accessible archive. It respects `robots.txt`, uses conservative request delays, and identifies itself to the server operator.

Publications using data downloaded with this tool should cite KCAC as the source:

```text
Kurdistan Center for Arts and Culture Digital Archive.
https://archive.kcac.org
```

Do not redistribute downloaded images or PDFs commercially.
