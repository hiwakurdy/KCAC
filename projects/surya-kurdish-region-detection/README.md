# Kurdish Sorani Text Region Detection

This project builds a local document-region detection workflow for Kurdish Sorani page images. It can:

- run Surya OCR/layout detection on unannotated page images;
- export detected text lines and layout regions as COCO JSON;
- export one compact JSON file per page for correction or training;
- create JPG visualizations for manual quality review;
- train a local Surya-style region detector from corrected per-image JSON;
- run the trained detector on new page folders and export COCO predictions.

The main goal is dataset creation: detect text-line and page-layout boxes automatically, review/correct them, then use them for OCR, annotation, or detector training.

## Project Layout

```text
surya_kurdish/
|-- README.md
|-- requirements.txt
|-- config.py
|-- run_pipeline.py
|-- plan_b.py
|-- train_region_detector.py
|-- predict_region_detector.py
|-- images/
|   |-- page_0001.jpg
|   |-- ...
|-- annotations/
|   |-- page_0001.json
|   |-- ...
|-- results/
|   |-- annotations/
|   |-- logs/
|   |-- visualizations/
|   |-- trained_predictions/
|   `-- training/
|       `-- surya_like_region_detector/
|-- utils/
|   |-- evaluate.py
|   |-- export.py
|   |-- preprocess.py
|   |-- region_model.py
|   |-- visualize.py
|   `-- __init__.py
```

## Requirements

Install the base requirements from `requirements.txt`.

```bash
pip install -r requirements.txt
```

Base packages:

| Package | Purpose |
|---|---|
| `surya-ocr` | Surya text-line and layout detection. |
| `transformers>=4.56.1,<5` | Required by the Surya model loader used here. |
| `torch`, `torchvision` | GPU/CPU inference and Faster R-CNN training. |
| `Pillow` | Image loading and RGB conversion. |
| `opencv-python` | Denoise, binarize, deskew, and draw boxes. |
| `numpy` | Image arrays and metric calculations. |
| `matplotlib` | Three-panel Surya visualization images. |
| `tqdm` | Progress bars. |
| `shapely`, `pandas` | Utility dependencies for OCR/data workflows. |

Optional packages:

| Package | Used By | Purpose |
|---|---|---|
| `paddlepaddle`, `paddleocr` | `plan_b.py --engine paddle` | Fallback text-line detection. |
| `python-doctr[torch]` | `plan_b.py --engine doctr` | Fallback text-line detection. |
| `easyocr` | `plan_b.py --engine easyocr` | Fallback text-line detection. |
| `label-studio` | manual review | Correct exported boxes in a UI. |

GPU is recommended. The existing training run used CUDA on an NVIDIA GeForce RTX 3090.

## Input Data

Put page images in `surya_kurdish/images/` or pass another folder with `--image-dir`.

Supported image extensions:

```text
.jpg .jpeg .png .tif .tiff .bmp .webp
```

Images are processed in sorted filename order. A common naming pattern is:

```text
page_0001.jpg
page_0002.jpg
page_0003.jpg
```

## Quick Start

From the repository root:

```bash
cd surya_kurdish
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run a small smoke test:

```bash
python run_pipeline.py --sample 10
```

Run the full Surya auto-annotation pipeline:

```bash
python run_pipeline.py
```

Run with per-image JSON export, which is the easiest format for later training:

```bash
python run_pipeline.py --per-image-json
```

## What `run_pipeline.py` Does

`run_pipeline.py` is the Surya bootstrap pipeline.

1. Collects input images from `--image-dir`.
2. Optionally preprocesses each image:
   - converts to RGB;
   - upscales small images to at least 1200 px width;
   - denoises with OpenCV fast non-local means;
   - estimates skew with a binary mask and Hough lines;
   - deskews when the estimated angle is at least 0.5 degrees.
3. Loads Surya text detection and layout detection models.
4. Runs batched detection.
5. Computes per-image quality metrics:
   - accepted text-line count;
   - average confidence;
   - maximum confidence;
   - text coverage percentage;
   - layout type counts.
6. Writes visualizations, COCO JSON, logs, and optional per-image JSON.
7. Prints a dataset-level decision: `PROCEED`, `PREPROCESS`, or `SWITCH`.

## Main Pipeline Options

| Option | Default | Description |
|---|---:|---|
| `--image-dir` | `images/` | Input image folder. |
| `--output-dir` | `results/` | Output folder for annotations, logs, and visualizations. |
| `--sample N` | none | Process only the first `N` sorted images. |
| `--conf FLOAT` | `0.40` | Minimum text-line confidence for export and quality metrics. |
| `--batch-size N` | `8` | Outer image batch size for Surya inference. |
| `--no-preprocess` | off | Skip upscale, denoise, and deskew. |
| `--no-viz` | off | Skip JPG visualization panels. |
| `--no-coco` | off | Skip combined COCO export. |
| `--per-image-json` | off | Also write one compact JSON file per image. |

Examples:

```bash
python run_pipeline.py --sample 20 --conf 0.30
python run_pipeline.py --image-dir E:\path\to\pages --output-dir results\dataset_409 --per-image-json
python run_pipeline.py --no-preprocess --sample 20
python run_pipeline.py --batch-size 4
```

## Surya Pipeline Outputs

Default output root: `surya_kurdish/results/`.

| Path | Created By | Contents |
|---|---|---|
| `results/visualizations/` | `run_pipeline.py` | Three-panel JPGs: original image, text lines, layout regions. |
| `results/annotations/coco_annotations_YYYYMMDD_HHMMSS.json` | `run_pipeline.py` | Combined COCO dataset. |
| `results/annotations/per_image/*.json` | `run_pipeline.py --per-image-json` | One compact annotation JSON per image. |
| `results/logs/pipeline.log` | `run_pipeline.py` | Runtime log. |
| `results/logs/report_YYYYMMDD_HHMMSS.json` | `run_pipeline.py` | Summary and per-image quality metrics. |

## Quality Decision Gate

The thresholds live in `config.py`.

| Flag | Criteria |
|---|---|
| `GOOD` | `avg_confidence >= 0.70`, at least `3` accepted text lines, and text coverage at least `5%`. |
| `MEDIUM` | `avg_confidence >= 0.50` and at least `1` accepted text line. |
| `POOR` | Anything below the `MEDIUM` gate. |

Dataset-level decisions:

| Decision | Criteria | Meaning |
|---|---|---|
| `PROCEED` | At least `70%` of images are `GOOD`. | Use the COCO/per-image JSON for annotation review or training. |
| `PREPROCESS` | At least `50%` of images are `GOOD` or `MEDIUM`. | Inspect visualizations, lower `--conf`, or compare preprocessing settings. |
| `SWITCH` | Too few usable images. | Run `plan_b.py` and compare fallback OCR engines. |

## COCO Output Format

Combined COCO JSON files contain:

```json
{
  "info": {
    "description": "Kurdish Sorani Surya Text Regions",
    "version": "1.0",
    "year": 2026,
    "contributor": "Surya Kurdish local auto-annotation pipeline",
    "date_created": "YYYY-MM-DDTHH:MM:SS"
  },
  "licenses": [
    {
      "id": 0,
      "name": "Unknown or user-provided",
      "url": ""
    }
  ],
  "categories": [],
  "images": [],
  "annotations": []
}
```

Each image record:

```json
{
  "id": 1,
  "file_name": "page_0001.jpg",
  "width": 1733,
  "height": 2480
}
```

Each annotation record uses COCO `xywh` boxes:

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [358.0, 1396.0, 1138.0, 81.0],
  "area": 92178.0,
  "segmentation": [],
  "iscrowd": 0,
  "attributes": {
    "confidence": 1.0,
    "source": "surya_detection"
  }
}
```

For Surya layout boxes, `attributes` use `score`, `surya_label`, and `source: surya_layout`.

## Per-Image JSON Format

The per-image format is simpler than COCO and is the input format used by `train_region_detector.py`.

```json
{
  "file": "page_0001.jpg",
  "width": 1733,
  "height": 2480,
  "bbox_format": "coco_xywh",
  "text_lines": [
    {
      "bbox": [358, 1396, 1138, 81],
      "confidence": 1.0
    }
  ],
  "layout": [
    {
      "bbox": [574, 2222, 730, 100],
      "label": "PageFooter",
      "score": 0.9892
    }
  ]
}
```

Important details:

- `bbox_format` is `coco_xywh`: `[x, y, width, height]`.
- `text_lines` are always trained/exported as category `1`.
- `layout` labels are mapped to the local COCO categories in `config.py`.
- The JSON basename must match the image basename for training, for example `page_0001.jpg` and `page_0001.json`.

## Labels

The local detector uses 10 classes: one background class plus 9 exported categories.

| ID | Name | Source |
|---:|---|---|
| 0 | `__background__` | Torchvision detector background. |
| 1 | `text_line` | Surya text detection or trained detector. |
| 2 | `paragraph` | Surya layout labels such as `Text`, `Paragraph`, `Text-inline-math`. |
| 3 | `title` | Surya layout labels such as `Title`, `Section-header`, `Heading`. |
| 4 | `table` | Surya layout `Table`. |
| 5 | `figure` | Surya layout labels such as `Figure`, `Picture`, `Image`. |
| 6 | `caption` | Surya layout labels such as `Caption`, `Footnote`. |
| 7 | `list_item` | Surya layout labels such as `List-item`, `List`. |
| 8 | `page_header` | Surya layout `Page-header`. |
| 9 | `page_footer` | Surya layout `Page-footer`. |

## Training A Local Region Detector

`train_region_detector.py` trains a Faster R-CNN ResNet-50 FPN detector on the per-image JSON format above. It can train text lines only, layout only, or both.

Default training paths in the script:

```text
train images:       E:\Antigravity_Code\get_pdfs_KCAC\kcac_client\dataset\409\pages
train annotations:  E:\Antigravity_Code\get_pdfs_KCAC\kcac_client\dataset\409\json_of_pages\annotations\per_image
val images:         E:\Antigravity_Code\get_pdfs_KCAC\kcac_client\dataset\409\test\img
val annotations:    E:\Antigravity_Code\get_pdfs_KCAC\kcac_client\dataset\409\test\anno
```

Training command:

```bash
python train_region_detector.py --epochs 12 --batch-size 2 --target both
```

Train on the local `surya_kurdish/images` and `surya_kurdish/annotations` folders:

```bash
python train_region_detector.py ^
  --train-image-dir images ^
  --train-anno-dir annotations ^
  --val-image-dir images ^
  --val-anno-dir annotations ^
  --epochs 75 ^
  --batch-size 2 ^
  --target both ^
  --resume results\training\surya_like_region_detector\best_model.pth
```

Smoke test:

```bash
python train_region_detector.py --max-train-images 8 --max-val-images 4 --epochs 1 --batch-size 1
```

Resume training:

```bash
python train_region_detector.py --resume results\training\surya_like_region_detector\last_model.pth --epochs 100
```

Evaluation only:

```bash
python train_region_detector.py ^
  --resume results\training\surya_like_region_detector\best_model.pth ^
  --eval-only
```

Training options:

| Option | Default | Description |
|---|---:|---|
| `--train-image-dir` | dataset `409/pages` | Training image folder. |
| `--train-anno-dir` | dataset `409/json_of_pages/annotations/per_image` | Training per-image JSON folder. |
| `--val-image-dir` | dataset `409/test/img` | Validation image folder. |
| `--val-anno-dir` | dataset `409/test/anno` | Validation per-image JSON folder. |
| `--out-dir` | `results/training/surya_like_region_detector` | Checkpoints, history, and log output folder. |
| `--epochs` | `12` | Last epoch number to train to. |
| `--batch-size` | `2` | Training batch size. |
| `--workers` | `0` | DataLoader worker count. |
| `--lr` | `0.0025` | SGD learning rate. |
| `--weight-decay` | `0.0001` | SGD weight decay. |
| `--momentum` | `0.9` | SGD momentum. |
| `--min-size` | `800` | Faster R-CNN min resize. |
| `--max-size` | `1333` | Faster R-CNN max resize. |
| `--score-thresh` | `0.35` | Validation prediction score threshold. |
| `--iou-thresh` | `0.50` | Validation IoU matching threshold. |
| `--min-conf` | `0.40` | Minimum text-line confidence loaded from training JSON. |
| `--min-layout-score` | `0.30` | Minimum layout score loaded from training JSON. |
| `--target` | `both` | `text_lines`, `layout`, or `both`. |
| `--max-train-images` | none | Limit training images for quick tests. |
| `--max-val-images` | none | Limit validation images for quick tests. |
| `--seed` | `42` | Split/reproducibility seed. |
| `--no-pretrained` | off | Do not initialize from COCO Faster R-CNN weights. |
| `--no-amp` | off | Disable CUDA mixed precision. |
| `--resume` | none | Checkpoint to resume or evaluate. |
| `--eval-only` | off | Evaluate `--resume` without training. |

Training outputs:

| File | Purpose |
|---|---|
| `results/training/surya_like_region_detector/best_model.pth` | Best validation F1 checkpoint. |
| `results/training/surya_like_region_detector/last_model.pth` | Latest checkpoint after each epoch. |
| `results/training/surya_like_region_detector/history.json` | Epoch metrics for the current/resumed run. |
| `results/training/surya_like_region_detector/history copy.json` | Earlier 25-epoch training history kept in this workspace. |
| `results/training/surya_like_region_detector/train.log` | Full training log. |
| `results/training/surya_like_region_detector/eval_report.json` | Created by `--eval-only`. |

## Current Training Results In This Workspace

The current `train.log` shows several runs. The latest completed run resumed from the best checkpoint and trained local annotations through epoch 75.

| Metric | Value |
|---|---:|
| Training target | `both` |
| Training image/json pairs | `42` |
| Validation image/json pairs | `42` |
| Device | `cuda` |
| GPU | `NVIDIA GeForce RTX 3090` |
| Best epoch | `67` |
| Best validation F1 | `0.9905` |
| Best validation precision | `0.9853` |
| Best validation recall | `0.9958` |
| Final epoch | `75` |
| Final epoch validation F1 | `0.9843` |
| Final epoch loss | `0.1981` |

The previous `history copy.json` covers an earlier 25-epoch run:

| Metric | Value |
|---|---:|
| First epoch | `1` |
| Last epoch | `25` |
| Best epoch | `24` |
| Best validation F1 | `0.9500` |
| Best validation precision | `0.9223` |
| Best validation recall | `0.9794` |

## Predicting With The Trained Detector

Use `predict_region_detector.py` with a saved checkpoint.

```bash
python predict_region_detector.py ^
  --checkpoint results\training\surya_like_region_detector\best_model.pth ^
  --image-dir E:\path\to\new\pages ^
  --out-dir results\trained_predictions\new_pages ^
  --per-image-json
```

Prediction options:

| Option | Default | Description |
|---|---:|---|
| `--checkpoint` | required | `.pth` model checkpoint to load. |
| `--image-dir` | required | Folder of images to predict. |
| `--out-dir` | `results/trained_predictions` | Output folder. |
| `--score` | `0.35` | Minimum detection score to export. |
| `--min-size` | `800` | Faster R-CNN min resize. |
| `--max-size` | `1333` | Faster R-CNN max resize. |
| `--no-viz` | off | Skip prediction JPG visualizations. |
| `--per-image-json` | off | Write one compact prediction JSON per image. |

Prediction outputs:

| Path | Contents |
|---|---|
| `OUT_DIR/trained_predictions_YYYYMMDD_HHMMSS.json` | Combined COCO prediction file. |
| `OUT_DIR/per_image/*.json` | Optional compact per-image detections. |
| `OUT_DIR/visualizations/*_trained_pred.jpg` | JPGs with boxes, labels, and scores. |
| `OUT_DIR/predict.log` | Prediction log. |

Compact trained prediction JSON format:

```json
{
  "file": "page_0001.jpg",
  "width": 1733,
  "height": 2480,
  "bbox_format": "coco_xywh",
  "detections": [
    {
      "category_id": 1,
      "category_name": "text_line",
      "bbox": [1237.4, 1578.6, 168.9, 58.0],
      "score": 0.9991
    }
  ]
}
```

## Current Prediction Results In This Workspace

For `results/trained_predictions/local_images/trained_predictions_20260505_020304.json`:

| Item | Count |
|---|---:|
| Images | `42` |
| Total annotations | `955` |
| `text_line` | `743` |
| `paragraph` | `130` |
| `title` | `35` |
| `figure` | `8` |
| `page_footer` | `39` |

For `results/trained_predictions/local_images_410/trained_predictions_20260505_020916.json`:

| Item | Count |
|---|---:|
| Images | `381` |
| Total annotations | `18220` |
| `text_line` | `16346` |
| `paragraph` | `1708` |
| `title` | `91` |
| `figure` | `10` |
| `page_footer` | `65` |

There are also 42 reviewed/usable per-image annotation JSON files in `surya_kurdish/annotations/`.

## Plan B Fallback OCR

Use `plan_b.py` when Surya misses too many text regions or the decision gate says `SWITCH`.

```bash
python plan_b.py --engine paddle --sample 20 --conf 0.35
python plan_b.py --engine doctr --sample 20
python plan_b.py --engine easyocr --sample 20
```

Plan B options:

| Option | Default | Description |
|---|---:|---|
| `--engine` | `paddle` | `paddle`, `doctr`, or `easyocr`. |
| `--sample` | none | Process only the first `N` sorted images. |
| `--conf` | `0.35` | Minimum confidence for exported boxes. |
| `--image-dir` | `images/` | Input image folder. |

Plan B outputs:

| Path | Contents |
|---|---|
| `results/annotations/plan_b_ENGINE_coco_YYYYMMDD_HHMMSS.json` | COCO text-line annotations. |
| `results/visualizations/plan_b_ENGINE/*_planb.jpg` | Plan B visualizations. |
| `results/logs/plan_b.log` | Plan B runtime log. |

Plan B exports only text-line boxes with `category_id: 1`.

## Manual Review With Label Studio

Install and start Label Studio:

```bash
pip install label-studio
label-studio start
```

Suggested workflow:

1. Create an object-detection project with bounding boxes.
2. Import a COCO file from `results/annotations/` or `results/trained_predictions/`.
3. Correct missing, shifted, or duplicate boxes.
4. Export corrected annotations.
5. Convert or save them into the per-image JSON format if training with `train_region_detector.py`.

## Practical Workflow

For a new page collection:

```bash
cd surya_kurdish
python run_pipeline.py --image-dir E:\path\to\pages --output-dir results\new_pages --per-image-json
```

Then inspect:

```text
results/new_pages/visualizations/
results/new_pages/logs/report_*.json
```

If the Surya result is good, review/correct the boxes and train:

```bash
python train_region_detector.py ^
  --train-image-dir E:\path\to\pages ^
  --train-anno-dir results\new_pages\annotations\per_image ^
  --epochs 12 ^
  --batch-size 2 ^
  --target both
```

Then predict on more pages:

```bash
python predict_region_detector.py ^
  --checkpoint results\training\surya_like_region_detector\best_model.pth ^
  --image-dir E:\path\to\more_pages ^
  --out-dir results\trained_predictions\more_pages ^
  --per-image-json
```

## Troubleshooting

| Problem | Try |
|---|---|
| Surya model load fails with `transformers>=5` | Reinstall with `python -m pip install "transformers>=4.56.1,<5" --force-reinstall`. |
| CUDA is not used | Check PyTorch CUDA install, then set `TORCH_DEVICE=cuda` before running the pipeline. |
| GPU memory error during Surya inference | Lower `--batch-size` to `4`, `2`, or `1`. |
| GPU memory error during training | Lower `--batch-size` to `1`; keep `--workers 0` on Windows. |
| Too few boxes are exported | Lower `--conf` for Surya or `--score` for trained prediction. |
| Too many low-quality boxes | Raise `--conf`, `--score`, `--min-conf`, or `--min-layout-score`. |
| Preprocessing hurts detection | Compare with `python run_pipeline.py --no-preprocess --sample 20`. |
| Surya output is weak overall | Run `plan_b.py` with `paddle`, `doctr`, and `easyocr`, then compare visualizations. |
| Training cannot find data | Confirm image and JSON basenames match exactly. |

## Notes

- Surya weights download on first use.
- Training checkpoints are large: the current `best_model.pth` and `last_model.pth` are about 330 MB each.
- COCO boxes and per-image boxes are exported as `[x, y, width, height]`.
- Torchvision internally trains with `[x1, y1, x2, y2]`; conversion happens in `utils/region_model.py`.
- Keep corrected annotations separate from raw auto-annotations when comparing quality.
