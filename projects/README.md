# Projects

This folder contains related KCAC research projects imported from local working directories.

## Included

| Folder | Source folder | Included files |
|---|---|---|
| `tesseract-ocr-training/` | `E:\Antigravity_Code\tesseract` | README, training/evaluation scripts, small reports, helper notebook |
| `kcac-pdf-fetcher/` | `E:\Antigravity_Code\get_pdfs_KCAC` | README, KCAC client source, probes, prompts, config examples |
| `surya-kurdish-region-detection/` | `E:\Antigravity_Code\CRAFT\surya_kurdish` | README, pipeline scripts, training/prediction scripts, utilities |

## Excluded

Generated or heavy local artifacts are excluded from the GitHub repository:

- `.git/`, `.cache/`, `.venv/`, `__pycache__/`
- generated PDFs and page-image datasets
- `output/`, `outputs/`, `results/`, `probe_output/`, `probe_output_409/`
- training folders, trained weights, checkpoints, and downloaded model caches
- local image and annotation folders

Keep those artifacts on disk, publish them as GitHub Releases if appropriate, or move citable datasets/models to Hugging Face or Zenodo.
