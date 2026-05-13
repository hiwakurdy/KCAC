# Linear Backlog: KCAC OCR Pipeline v0.1

Default project: `KCAC OCR Pipeline v0.1`

## Repo/bootstrap/config/Docker

- Create Python package, config loader, CLI, Dockerfile, and reproducible requirements.
- Add `doctor` checks for all local binaries, Python packages, model paths, and API keys.

## Engine Adapters And Smoke Checks

- Implement Tesseract, Kraken, Calamari, Qwen2-VL, and Claude adapters.
- Run one-page smoke checks with all enabled engines.

## Unicode Normalisation

- Implement Sorani mappings, NFC composition, tashkeel stripping, and trace logs.
- Add explicit before/after tests.

## Consensus And Geometry Alignment

- Implement polygon IoU alignment, edit-distance voting, confidence labels, and failure preservation.

## PAGE XML And eScriptorium Export

- Export PAGE XML 2019-07-15 with raw and normalised `TextEquiv`.
- Build direct eScriptorium sidecar import layout.

## Queue, Reports, Benchmark

- Produce annotation queue, histogram, disagreement matrix, coverage matrix, daily summary, baseline metrics, and HF JSONL.

## Tests, Docs, Release Packaging

- Keep pytest, ruff, and mypy quality gates green.
- Publish setup, troubleshooting, normalisation, and schema compliance docs.
