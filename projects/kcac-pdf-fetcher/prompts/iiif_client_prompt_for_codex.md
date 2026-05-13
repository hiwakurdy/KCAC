# IIIF Digital Library Client — Building a Research Dataset

I am a PhD researcher in AI/ML working on **Kurdish-language Optical Character
Recognition (OCR)**. As part of my dissertation, I am building a pilot training
dataset from publicly accessible digital libraries that publish their materials
through standard library protocols (IIIF, DeepZoom).

I need a Python tool that **consumes standard IIIF Image API and IIIF
Presentation API endpoints** to retrieve high-resolution page images from
digital library books, then assembles them into PDFs suitable for OCR research.

The library platform I am working with is **MediaINFO Digital Library**, which
implements the IIIF specification. IIIF (International Image Interoperability
Framework) is the standard protocol used by Harvard, the British Library, the
Library of Congress, the Bavarian State Library, Gallica/BnF, and most major
research libraries worldwide. It is designed for exactly this use case:
programmatic access for scholars.

## What I need

### Component 1 — `iiif_discover.py`
A small utility that, given a book viewer URL, finds the IIIF manifest URL by:

1. Loading the page with **Playwright** (headless Chromium)
2. Watching for network requests that match the documented IIIF patterns:
   - `info.json` (IIIF Image API)
   - `manifest.json` or `manifest` (IIIF Presentation API)
   - URLs matching the IIIF Image API URL template:
     `{scheme}://{server}/{prefix}/{identifier}/{region}/{size}/{rotation}/{quality}.{format}`
3. Printing the discovered manifest URL
4. If no IIIF endpoint is found, reporting that fact and exiting (the script
   should NOT attempt to handle non-IIIF protocols)

### Component 2 — `iiif_fetch.py`
A standard IIIF client that, given a manifest URL, downloads each canvas
(page) at maximum resolution using the documented IIIF Image API:

- **IIIF v2**: `{service_id}/full/full/0/default.jpg`
- **IIIF v3**: `{service_id}/full/max/0/default.jpg`

For each book it should produce:

```
./output/{book_id}/
├── metadata.json                 // title, author, page_count, manifest_url
├── pages/page_0001.jpg, ...      // full-resolution images, no recompression
└── {book_id}.pdf                 // assembled with img2pdf (lossless)
```

### Component 3 — `validate.py`
After running, verify the output: count pages, check image integrity with
Pillow, confirm PDF page count matches.

## Required behavior (research-grade tooling)

1. **Polite request rate** — 2 seconds between page requests, 5 seconds
   between books, single connection (no parallelism). This matches the
   guidance in the IIIF community's published best practices for bulk usage.
2. **Resumable** — if a page file already exists and is valid, skip it.
3. **Identifying User-Agent** — `Kurdish-OCR-Research/1.0 (PhD dissertation;
   contact: <my email>)` so the library administrators can identify the
   traffic and contact me if anything is wrong.
4. **Respect `robots.txt`** — check on first run, exit with error if the
   relevant paths are disallowed.
5. **Graceful backoff** — on HTTP 429 or 503, exponential backoff
   (10s, 30s, 90s, 270s), then stop.
6. **Per-page error tolerance** — retry 3× with backoff; on persistent
   failure, log and continue to the next page.
7. **Cite the source** — every `metadata.json` records the source URL and a
   `license_note` reminding me to cite the library in any publication.

## Inputs

```bash
# Discovery (run first, once per library):
python iiif_discover.py --viewer-url https://archive.kcac.org/zoom/399/view

# Bulk fetch (after discovery confirms IIIF):
python iiif_fetch.py \
  --manifest-urls manifests.txt \
  --output ./dataset \
  --delay 2.0 \
  --book-delay 5.0
```

`manifests.txt` is one manifest URL per line — produced by running
`iiif_discover.py` for each book and collecting the outputs.

## Code requirements
- Python 3.10+
- Type hints throughout (`from __future__ import annotations`)
- Dependencies: `playwright`, `requests`, `Pillow`, `img2pdf`, `tqdm`
- Pinned in `requirements.txt`
- Standard `argparse` CLI, `--help` for each script
- Clear `README.md` with setup and usage examples
- Logging: console INFO, file DEBUG

## Reference specifications
- IIIF Image API 3.0: <https://iiif.io/api/image/3.0/>
- IIIF Presentation API 3.0: <https://iiif.io/api/presentation/3.0/>
- IIIF community best practices: <https://iiif.io/community/>
- img2pdf (lossless PDF assembly): <https://gitlab.mister-muffin.de/josch/img2pdf>

## Constraints
- This tool is **only for IIIF-compliant endpoints**. If a library does not
  expose IIIF, the script should report that and exit cleanly.
- The tool must not attempt any form of authentication, paywall circumvention,
  or DRM handling. IIIF endpoints are designed to be openly accessible by
  policy of the publishing institution.
- Output is for personal academic use; my publications will cite the source
  library.

Please start with `iiif_discover.py`. Once I confirm the discovery works on
my target library, write `iiif_fetch.py`.
