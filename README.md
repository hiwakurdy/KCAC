# KCAC Research Monorepo

This repository collects the local KCAC OCR and document-understanding projects into one GitHub repository.

The main project at the repository root is the KCAC OCR Pipeline v0.1. Additional related tools live under `projects/`.

## Projects

| Path | Purpose | Main documentation |
|---|---|---|
| `.` | OCR pipeline for formatting datasets, testing OCR/VLM engines, consensus, PAGE XML, eScriptorium export, reports, and Hugging Face JSONL export. | `docs/how_to_run_and_use.md` |
| `projects/tesseract-ocr-training/` | Tesseract, PaddleOCR, and OCR baseline training/evaluation scripts for Kurdish OCR experiments. | `projects/tesseract-ocr-training/README.ne` |
| `projects/kcac-pdf-fetcher/` | KCAC archive downloader/client for page images, PDFs, OCR text extraction, and validation. | `projects/kcac-pdf-fetcher/README.ne` |
| `projects/surya-kurdish-region-detection/` | Surya-based Kurdish text-line and layout-region detection workflow. | `projects/surya-kurdish-region-detection/README.md` |

Large generated artifacts are intentionally not part of git: PDFs, page-image datasets, caches, trained weights, model checkpoints, output folders, and annotation outputs. Keep those locally or publish them separately as dataset/model releases.

## OCR Pipeline Quick Start

Run from the repository root:

```powershell
(& conda shell.powershell hook) | Out-String | Invoke-Expression
conda activate surya
python -m pip install -r requirements.txt
python -m pipeline --config config.yaml.example doctor
```

For your RTX 3090 and Ollama Qwen setup:

```powershell
ollama list
ollama run qwen25vl-sorani-ocr:latest
python -m pipeline --config config.ollama.example doctor
python -m pipeline --config config.ollama.example bootstrap --limit 1
```

The full setup guide, file map, and reasoning are in `docs/how_to_run_and_use.md`.

## OCR Pipeline Commands

```powershell
python -m pipeline --config config.yaml bootstrap --limit 1
python -m pipeline --config config.yaml consensus --limit 1
python -m pipeline --config config.yaml pagexml --limit 1
python -m pipeline --config config.yaml escriptorium --limit 1
python -m pipeline --config config.yaml queue
python -m pipeline --config config.yaml reports
python -m pipeline --config config.yaml benchmark
python -m pipeline --config config.yaml hf-export
```

`run-all` runs the same stages in order. Existing per-engine/page outputs are skipped unless `--force` is passed.

## Model And Data Notes

Kraken is the optional in-process legacy OCR dependency in `requirements-ocr-py310.txt` because it is not installable in Python 3.13. Calamari is split into `requirements-calamari-py310.txt` and should be installed in a separate environment because its `python-bidi` dependency conflicts with Kraken.

Use `config.full.example` only after you place real checkpoints under `models/kraken/` and `models/calamari/` or edit those paths. `doctor` is expected to fail for any enabled engine whose binary, checkpoint, Ollama server/model, or API key is missing.
