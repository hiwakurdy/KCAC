# KCAC Research Monorepo

This repository collects the local KCAC OCR and document-understanding projects into one GitHub repository.

The main project at the repository root is the KCAC OCR Pipeline v0.1. Additional related tools live under `projects/`.

## Projects

| Path | Purpose | Main documentation |
|---|---|---|
| `.` | OCR pipeline — engines, consensus, PAGE XML, eScriptorium export, reports, Hugging Face export. | `docs/how_to_run_and_use.md` |
| `workbench/` | Browser-based annotation editor for page images and JSON output. | Open `workbench/home.html` in a browser. |
| `docs/plans/` | Product specification, critical review, and planning notes. | `docs/plans/v1.md` |
| `projects/tesseract-ocr-training/` | Tesseract, PaddleOCR, and OCR baseline training/evaluation scripts. | `projects/tesseract-ocr-training/README.ne` |
| `projects/kcac-pdf-fetcher/` | KCAC archive downloader/client for page images, PDFs, OCR text, and validation. | `projects/kcac-pdf-fetcher/README.ne` |
| `projects/surya-kurdish-region-detection/` | Surya-based Kurdish text-line and layout-region detection workflow. | `projects/surya-kurdish-region-detection/README.md` |

Large generated artifacts are intentionally not part of git: PDFs, page-image datasets, caches, trained weights, model checkpoints, output folders, and annotation outputs. Keep those locally or publish them separately as dataset/model releases.

## OCR Pipeline Quick Start

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. Copy and edit the config for your setup
cp config.yaml.example config.yaml   # minimal: Tesseract only, no GPU
# OR
cp config.ollama.example config.yaml  # Ollama + RTX 3090
# OR
cp config.full.example config.yaml    # all five engines

# 3. Check your environment
python -m pipeline doctor             # auto-detects config.yaml

# 4. One-page smoke test
python -m pipeline bootstrap --limit 1
```

On Windows with conda, activate your environment first:
```powershell
(& conda shell.powershell hook) | Out-String | Invoke-Expression
conda activate surya
```

See `docs/how_to_run_and_use.md` for the full setup guide. Run `make help` for a list of convenience targets.

## OCR Pipeline Commands

Once `config.yaml` exists in the repo root, `--config` can be omitted:

```powershell
python -m pipeline bootstrap --limit 1   # run OCR engines, write raw JSON
python -m pipeline consensus --limit 1   # align + vote across engines
python -m pipeline pagexml   --limit 1   # export PAGE XML for eScriptorium
python -m pipeline escriptorium          # build eScriptorium import folder
python -m pipeline queue                 # build human-review priority list
python -m pipeline reports               # agreement reports and summaries
python -m pipeline benchmark             # CER/WER after ground-truth exists
python -m pipeline hf-export             # Hugging Face JSONL line dataset
python -m pipeline run-all               # all stages in order
```

`run-all` runs all stages in order. Existing per-engine/page outputs are skipped unless `--force` is passed.

## Annotation Workbench

Open `workbench/home.html` in a browser to inspect and correct page annotations interactively. Use the in-app file-open buttons to load images and JSON from `output/` or `ds_test/`.

## Model And Data Notes

Kraken is the optional in-process legacy OCR dependency in `requirements-ocr-py310.txt` because it is not installable in Python 3.13. Calamari is split into `requirements-calamari-py310.txt` and should be installed in a separate environment because its `python-bidi` dependency conflicts with Kraken.

Use `config.full.example` only after you place real checkpoints under `models/kraken/` and `models/calamari/` or edit those paths. `doctor` is expected to fail for any enabled engine whose binary, checkpoint, Ollama server/model, or API key is missing.
