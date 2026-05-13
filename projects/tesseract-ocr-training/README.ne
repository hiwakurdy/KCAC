# KCAC Central Kurdish Historical Print OCR and Document Understanding Project

This repository contains the working OCR benchmark, model-training, and result
generation code for a larger research project: building a research-grade
document understanding dataset for printed Central Kurdish (Sorani), anchored
on the Kurdistan Center for Arts and Culture (KCAC) archive.

The long-term goal is not only OCR. The project is designed as reusable
infrastructure for Kurdish document AI: OCR, layout analysis, clean text
corpora, language modelling, named-entity recognition, machine translation,
historical linguistics, full-text archive search, spell correction, dictionary
building, and digital humanities research.

## Project Goals

1. Build a public, citable dataset for historical Kurdish printed documents.
2. Produce verified OCR ground truth for Central Kurdish / Sorani print.
3. Train and evaluate OCR baselines using Tesseract, EasyOCR, PaddleOCR, and
   later stronger document AI models.
4. Define a clear annotation schema for page images, regions, lines, words,
   characters, metadata, and normalized text.
5. Return useful outputs to KCAC: searchable text, enriched metadata, OCR
   models, benchmark reports, and a reproducible pipeline.
6. Publish a technical report and dataset card so other researchers can reuse
   the data correctly.

## Why This Project Is Needed

Kurdish is spoken by tens of millions of people but remains under-resourced in
OCR and document AI. Public resources exist for translation, Wikipedia-style
text, sentiment, speech, and web corpora, but the major missing pieces are:

- no public OCR benchmark for historical Kurdish print;
- no public layout-analysis dataset for Kurdish books or periodicals;
- no public ground-truth transcription corpus for historical Sorani;
- no public historical document dataset aligned with PAGE XML, ALTO, TEI,
  IIIF, METS, JSONL, and Parquet export formats.

Existing OCR systems have not been trained on enough representative historical
Kurdish print. KCAC's archive is valuable because it covers many eras, genres,
print technologies, fonts, paper conditions, and source collections.

## Planned KCAC Dataset

Working project name:

```text
KCAC Central Kurdish Historical Print Document AI Dataset
```

Pilot phase target:

- 10-20 selected books;
- 1,000-2,000 fully annotated pages;
- coverage across historical period, typography, layout, genre, and scan
  quality;
- baseline OCR models for Sorani and Kurmanji;
- public dataset release, likely `v0.1`;
- published technical report;
- full-text search demonstration over the pilot books.

Expansion phase target:

- 100+ books;
- 10,000+ annotated pages;
- active learning loop for hard-page discovery;
- dataset `v1.0`;
- broader contribution from Kurdish linguists and archive partners.

Scale phase target:

- larger KCAC archive coverage;
- hosted search portal;
- repeatable benchmark and shared-task style evaluation;
- long-term citable dataset snapshots through Hugging Face Hub and Zenodo.

## Dataset Architecture

The dataset should be modular. Each layer can be cited, downloaded, and used
separately.

| Layer | Name | Main use |
|---|---|---|
| L1 | Bibliographic catalog | Browse and search books before OCR |
| L2 | Page image collection | Preservation and computer vision |
| L3 | OCR ground-truth corpus | OCR training, evaluation, and search |
| L4 | Layout and structure dataset | Page segmentation, reading order, regions |
| L5 | NLP research corpus | Language modelling, NER, dictionary, history |

## Bibliographic Metadata Format

Each book should have stable metadata. Recommended fields:

| Field | Meaning |
|---|---|
| `book_id` | Stable ID, for example `kcac_000152` |
| `title_original` | Title exactly as printed |
| `title_normalised` | Modern normalized title, optional |
| `authors` | Author names as printed |
| `authors_normalised` | Standardized names and authority IDs |
| `editor`, `compiler`, `translator` | Contributors when applicable |
| `publisher` | Publisher or printing house |
| `place_of_publication` | City or place |
| `publication_year_printed` | Year as printed |
| `publication_year_gregorian` | Gregorian year |
| `publication_year_hijri` | Hijri year if present |
| `publication_year_kurdish` | Kurdish year if present |
| `edition` | Edition number |
| `language` | BCP 47 code, for example `ckb`, `kmr`, `hac` |
| `script` | Script code, for example `Arab` or `Latn` |
| `genre` | Controlled genre label |
| `tags` | Additional subject tags |
| `rights_status` | Public domain, permission granted, restricted, unknown |
| `source_collection` | KCAC source collection |
| `source_url` | Permanent archive URL |
| `physical_condition` | pristine, good, damaged, fragmentary |
| `missing_pages` | Known missing pages |
| `notes` | Damage, typography, uncertain dates, special remarks |

## Annotation Schema

The full dataset should support several levels of annotation.

### Image Level

Recommended image-level fields:

- `scan_resolution_dpi`
- `pixel_dimensions`
- `colour_mode`
- `compression`
- `binding_visible`
- `scan_skew_degrees`
- `scan_quality`
- `damage_present`
- `blur_score`
- `contrast_score`
- `noise_score`

Recommended image derivatives:

| Derivative | Format | Purpose |
|---|---|---|
| `master` | TIFF | preservation |
| `cleaned` | PNG | OCR training and inference |
| `thumbnail` | JPEG | preview and browsing |
| `line_crops` | PNG | line OCR training |
| `word_crops` | PNG | word spotting and advanced OCR subset |

### Page Level

Recommended page-level fields:

- `page_number_printed`
- `page_number_sequence`
- `page_type`
- `layout_class`
- `reading_direction`
- `script`
- `era_typography`
- `dominant_font`
- `has_illustrations`
- `has_table`
- `has_marginalia`
- `has_footnote`
- `has_poetry`
- `verification_status`

### Region Level

Recommended region classes:

- `TextRegion`
- `Heading`
- `Subtitle`
- `AuthorLine`
- `PoetryRegion`
- `Verse`
- `Stanza`
- `Hemistich`
- `Caption`
- `Footnote`
- `PageNumber`
- `Header`
- `Footer`
- `Marginalia`
- `Illustration`
- `Figure`
- `Table`
- `DecorativeElement`
- `MathRegion`
- `Signature`
- `Stamp`
- `Seal`
- `Advertisement`
- `DamagedRegion`

Each region should include a polygon and a reading-order index.

### Line Level

Recommended line-level fields:

| Field | Meaning |
|---|---|
| `line_id` | Stable line ID, for example `kcac_000152_p0007_l0032` |
| `polygon` | Line polygon |
| `baseline` | Baseline points |
| `bbox` | Axis-aligned bounding box |
| `reading_order` | Order inside region |
| `text_raw` | Diplomatic transcription |
| `text_normalised` | Modern normalized text |
| `is_handwritten` | Boolean |
| `font_style` | normal, bold, italic, decorative, handwritten |
| `damage_local` | faded, torn, stained, blurred, etc. |
| `quality` | verified, double_verified, uncertain |
| `language_local` | Per-line language tag |

### Word and Character Level

Word boxes should be produced for a sample subset. Character boxes should be
created only for a smaller gold benchmark subset, around 50-100 pages.

## Text Normalization

The project should preserve several parallel text layers:

1. `text_raw`: exactly as printed.
2. `text_normalised`: modern Sorani orthography.
3. `text_diacritic_stripped`: useful for search and some OCR experiments.
4. `text_latin_transliteration`: optional subset.
5. `translation`: only where source material already contains parallel text.

Current benchmark normalization includes:

- Unicode NFC normalization;
- ZWNJ and ZWJ removal;
- whitespace collapse;
- Arabic/Persian Yeh and Kaf folding: `ي -> ی`, `ك -> ک`.

The Tesseract fine-tuning script also records character changes in:

```text
train/urd/normalization_report.txt
```

Current report:

```text
4359 map U+060C ، ARABIC COMMA -> ,
123  map U+06A4 ڤ ARABIC LETTER VEH -> ف
33   map U+0640 ـ ARABIC TATWEEL -> <drop>
1    map U+2044 ⁄ FRACTION SLASH -> /
```

## Data Split Rules

For real KCAC releases, split by book, not by page. Pages from the same book
share font, scanner, paper, and layout properties, so page-level splitting
would make evaluation too easy.

Recommended test sets:

| Test set | Purpose |
|---|---|
| `test_general` | Balanced headline evaluation |
| `test_historical_hard` | older print, difficult typography |
| `test_low_quality_scans` | damaged, faded, blurred pages |
| `test_poetry_layout` | poetry layouts and dense marks |
| `test_multi_column` | periodicals, dictionaries, columns |
| `test_mixed_language` | Arabic, Persian, Turkish, Kurdish mixtures |
| `test_long_tail_glyphs` | rare or archaic characters |

## Quality Levels

Every annotation record should carry `verification_status`.

| Status | Meaning | Public release |
|---|---|---|
| `raw` | not checked | internal only |
| `ocr_generated` | machine OCR only | optional |
| `single_verified` | one human correction pass | yes, tagged |
| `double_verified` | two annotators agree | yes |
| `adjudicated` | expert resolved disagreement | yes |
| `gold` | final benchmark quality | yes |

## Quality Metrics

The project should publish these metrics:

- Character Error Rate, `CER`;
- Word Error Rate, `WER`;
- character precision, recall, and F1;
- word precision, recall, and F1;
- Layout F1;
- bounding-box IoU;
- reading-order accuracy;
- inter-annotator agreement;
- percentage of unreadable or damaged text;
- coverage matrix completeness by era, typography, layout, and genre.

## Repository Layout

```text
.
├── run_train_ara.ps1
├── run_train_urd.ps1
├── sync_ocr_result_files.py
├── scripts/
│   ├── finetune.py
│   ├── evaluate_tesseract_dataset.py
│   ├── evaluate_paddleocr_dataset.py
│   ├── prepare_paddle_rec_dataset.py
│   ├── train_paddle_rec.py
│   ├── train_paddleocr_rec_direct.py
│   ├── ocr_surya_annotations.py
│   └── ocr_surya_annotations_paddle.py
├── Testing/
│   ├── tesseract_kurdish_eval.py
│   ├── easyocr_kurdish_eval.py
│   ├── paddleocr_kurdish_eval.py
│   ├── combine_ocr_results.py
│   ├── make_kurdish_char_errors.py
│   ├── make_ocr_status.py
│   └── run_new_data_tesseract_easyocr_combine.ps1
├── train/
│   └── urd/
│       ├── ground-truth/
│       ├── lists/
│       ├── checkpoints/
│       ├── output/
│       └── metrics/
└── researcher/
    └── ocr_10k_results_report.md
```

## Input Dataset Format for Current Scripts

Most current scripts expect paired image and text files in one directory:

```text
dataset/
├── img_001.png
├── img_001.txt
├── img_002.png
├── img_002.txt
└── ...
```

Default dataset paths used by the scripts:

```text
E:\TRDG\new_ds_for_finetune\test_nrt_pdf_images
E:\TRDG\data\kurdish_tts_10k_multifont
```

Each `.txt` file is the ground-truth transcription for the image with the
same stem.

## Requirements

System tools:

- Windows PowerShell;
- Python 3.11+ or compatible conda environments;
- Tesseract OCR 5.x;
- Tesseract training tools: `lstmtraining`, `combine_tessdata`;
- installed Tesseract language data for `ara`, `fas`, and `urd`;
- CUDA-capable GPU for EasyOCR and PaddleOCR GPU runs, recommended.

Python packages used by the repository:

- `Pillow`;
- `easyocr`;
- `torch`;
- `paddlepaddle`;
- `paddleocr`;
- `paddlex`;
- `PyYAML`;
- standard-library modules such as `argparse`, `csv`, `json`, `subprocess`,
  `statistics`, `unicodedata`, and `pathlib`.

Known local environments referenced by scripts:

```text
C:\ProgramData\miniconda3\python.exe
C:\Users\Hiwa\.conda\envs\spade\python.exe
C:\Users\Hiwa\.conda\envs\paddleocr_eval\python.exe
```

The scripts also route Paddle cache files into the repo-local `.cache/`
directory to reduce Windows home-directory permission issues.

## Tesseract Fine-Tuning

The main fine-tuning script is:

```text
scripts/finetune.py
```

Convenience launchers:

```powershell
.\run_train_urd.ps1
.\run_train_ara.ps1
```

Default Urdu run:

```powershell
.\run_train_urd.ps1 `
  -Dataset "E:\TRDG\new_ds_for_finetune\test_nrt_pdf_images" `
  -Workdir "train\urd" `
  -BaseLang "script/Arabic" `
  -BaseTraineddata "train\base_models\script\Arabic.traineddata" `
  -MaxIterations 2000 `
  -Psm 7 `
  -FallbackPsm "13"
```

What it does:

1. Finds `.png` / `.txt` image-text pairs.
2. Splits data into train and eval subsets.
3. Stages Tesseract ground-truth files under `train/<lang>/ground-truth`.
4. Generates `.lstmf` files with Tesseract.
5. Extracts base LSTM components from traineddata.
6. Runs `lstmtraining`.
7. Writes the final model to:

```text
train/urd/output/urd.traineddata
```

Important training outputs:

| Path | Meaning |
|---|---|
| `train/urd/ground-truth/` | staged images, `.gt.txt`, `.box`, `.lstmf` |
| `train/urd/lists/urd.training_files.txt` | training `.lstmf` list |
| `train/urd/lists/urd.eval_files.txt` | eval `.lstmf` list |
| `train/urd/checkpoints/` | Tesseract checkpoints |
| `train/urd/output/urd.traineddata` | final trained model |
| `train/urd/train.log` | full Tesseract training log |
| `train/urd/normalization_report.txt` | character mapping/drop report |

Use the trained model:

```powershell
tesseract IMAGE stdout -l urd --tessdata-dir "train\urd\output"
```

## Tesseract Evaluation

Script:

```text
scripts/evaluate_tesseract_dataset.py
```

Example:

```powershell
python scripts\evaluate_tesseract_dataset.py `
  --dataset "E:\TRDG\new_ds_for_finetune\test_nrt_pdf_images" `
  --tessdata-dir "train\urd\output" `
  --lang urd `
  --output-dir "train\urd\metrics" `
  --split-name eval `
  --psm 7 `
  --dpi 300
```

Outputs:

| File or folder | Meaning |
|---|---|
| `<split>_summary.txt` | aggregate CER, WER, precision, recall, F1 |
| `<split>_per_file.csv` | per-image metrics |
| `predictions/<split>/*.hyp.txt` | OCR prediction text per image |

Current trained `urd` evaluation result on `test_nrt_pdf_images`:

```text
files: 225
CER_percent: 92.8284
WER_percent: 99.0817
char_recall_percent: 7.1806
char_precision_percent: 56.0209
```

This result shows that the current fine-tuned model is not yet good enough
for production OCR. It is useful as a reproducible training experiment, but
the best current benchmark result comes from base OCR configurations on the
10,000-image benchmark.

## EasyOCR, PaddleOCR, and Tesseract Benchmark

Main benchmark orchestration script:

```text
Testing/run_new_data_tesseract_easyocr_combine.ps1
```

It runs:

1. Tesseract CPU for `ara`, `fas`, `urd`, mixed-language configs, and no-DAWG
   variants.
2. EasyOCR GPU for `ar`, `fa`, `ur`, and language-list combinations.
3. EasyOCR GPU for `ar+fa+ur`.
4. PaddleOCR PP-OCRv5 result syncing.
5. Kurdish-specific character error analysis.
6. Combined summary table generation.

Example:

```powershell
cd Testing
.\run_new_data_tesseract_easyocr_combine.ps1
```

Core scripts:

| Script | Purpose |
|---|---|
| `Testing/tesseract_kurdish_eval.py` | benchmark Tesseract language packs |
| `Testing/easyocr_kurdish_eval.py` | benchmark EasyOCR language settings |
| `Testing/paddleocr_kurdish_eval.py` | benchmark PaddleOCR Arabic-script settings |
| `Testing/combine_ocr_results.py` | combine `summary.csv` files |
| `Testing/make_kurdish_char_errors.py` | analyze Kurdish-specific character errors |
| `Testing/make_ocr_status.py` | create `status.txt` and `status.json` |

## Current 10,000-Image Benchmark Results

Dataset:

```text
E:\TRDG\data\kurdish_tts_10k_multifont
```

Combined result file:

```text
Testing/new_data/results_all_models/combined_summary.txt
```

Paper-ready result table:

| Configuration | Engine | CER % | WER % | Time s/image | Coverage % |
|---|---|---:|---:|---:|---:|
| Tesseract (ara) | Tesseract CPU | 37.63 | 89.73 | 0.12 | 98.3 |
| Tesseract (fas) | Tesseract CPU | 28.09 | 82.17 | 0.10 | 98.3 |
| Tesseract (urd) | Tesseract CPU | 20.33 | 58.75 | 0.12 | 98.3 |
| Tesseract (ara+fas) | Tesseract CPU | 31.63 | 87.41 | 0.16 | 98.3 |
| Tesseract (fas+urd) | Tesseract CPU | 26.29 | 77.92 | 0.17 | 98.3 |
| Tesseract (ara+urd) | Tesseract CPU | 29.08 | 76.65 | 0.19 | 98.3 |
| Tesseract (fas, no DAWG) | Tesseract CPU | 28.09 | 82.17 | 0.10 | 98.3 |
| Tesseract (ara, no DAWG) | Tesseract CPU | 37.63 | 89.73 | 0.12 | 98.3 |
| EasyOCR (ar) | EasyOCR GPU | 29.92 | 91.36 | 0.11 | 100.0 |
| EasyOCR (fa) | EasyOCR GPU | 24.81 | 85.21 | 0.13 | 100.0 |
| EasyOCR (ur) | EasyOCR GPU | 20.55 | 75.06 | 0.09 | 100.0 |
| EasyOCR (ar+fa) | EasyOCR GPU | 24.81 | 85.21 | 0.10 | 100.0 |
| EasyOCR (fa+ur) | EasyOCR GPU | 21.84 | 78.25 | 0.09 | 100.0 |
| EasyOCR (ar+ur) | EasyOCR GPU | 21.84 | 78.25 | 0.09 | 100.0 |
| EasyOCR (ar+fa+ur) | EasyOCR GPU | 21.84 | 78.25 | 0.09 | 100.0 |
| PaddleOCR v5 (ar) | PaddleOCR GPU PP-OCRv5 | 25.51 | 81.80 | 0.06 | 100.0 |
| PaddleOCR v5 (fa) | PaddleOCR GPU PP-OCRv5 | 25.51 | 81.80 | 0.07 | 100.0 |
| PaddleOCR v5 (ur) | PaddleOCR GPU PP-OCRv5 | 25.51 | 81.80 | 0.06 | 100.0 |

Best completed results:

```text
Best CER:     Tesseract (urd), 20.33%
Best WER:     Tesseract (urd), 58.75%
Fastest mean: PaddleOCR v5 (ar), 0.06 s/image
```

Main interpretation:

- Tesseract `urd` is the strongest overall setting on this synthetic
  10,000-image benchmark.
- EasyOCR `ur` is the strongest EasyOCR setting and is very close in CER to
  Tesseract `urd`.
- PaddleOCR PP-OCRv5 is fastest, but its CER/WER are weaker than the best
  Tesseract and EasyOCR rows.
- Multi-language settings do not improve over the best single-language
  setting.
- Disabling the Tesseract Farsi DAWG files did not change the measured Farsi
  score in this run.

## Result Directory Format

Every OCR result directory should contain as many of these files as possible:

| File or folder | Meaning |
|---|---|
| `summary.csv` | aggregate metrics per configuration |
| `summary.txt` | human-readable summary |
| `per_image.csv` | per-image CER, WER, time, coverage |
| `environment.txt` | package, device, and model details |
| `status.txt` | progress summary |
| `status.json` | machine-readable progress |
| `stats.txt` | extra statistics when available |
| `run.log` | captured command output |
| `kurdish_char_errors.csv` | Kurdish-specific character error report |
| `predictions/<config_id>/*.hyp.txt` | OCR predictions per image |

Combined benchmark directories contain:

| File | Meaning |
|---|---|
| `combined_summary.csv` | merged table across engines |
| `combined_summary.txt` | paper-ready table plus best rows |

## Combined Result Generation

Use:

```powershell
python Testing\combine_ocr_results.py `
  --out Testing\new_data\results_all_models `
  --input "Tesseract CPU=Testing\new_data\results_tesseract_cpu_all" `
  --input "EasyOCR GPU=Testing\new_data\results_easyocr_gpu_all" `
  --input "EasyOCR GPU=Testing\new_data\results_easyocr_ar_plus_fa_plus_ur" `
  --input "PaddleOCR GPU PP-OCRv5=Testing\new_data\results_paddleocr_gpu_v5"
```

To copy only core result files without copying large prediction trees:

```powershell
python sync_ocr_result_files.py `
  --src Testing\data002\results_paddleocr_gpu_v5 `
  --dst Testing\new_data\results_paddleocr_gpu_v5
```

## PaddleOCR Recognition Training

Prepare PaddleOCR recognition dataset:

```powershell
python scripts\prepare_paddle_rec_dataset.py `
  --dataset "E:\TRDG\new_ds_for_finetune\test_nrt_pdf_images" `
  --output-dir train\paddle_rec_dataset `
  --train-list train\ara\lists\ara.training_files.txt `
  --val-list train\ara\lists\ara.eval_files.txt
```

Output format:

```text
train/paddle_rec_dataset/
├── train.txt
├── val.txt
├── dict.txt
└── summary.txt
```

Each line in `train.txt` and `val.txt` uses PaddleOCR recognition format:

```text
path/to/image.png<TAB>label text
```

Direct PaddleOCR training:

```powershell
& "C:\Users\Hiwa\.conda\envs\paddleocr_eval\python.exe" `
  scripts\train_paddleocr_rec_direct.py `
  --dataset-dir train\paddle_rec_dataset `
  --output-dir train\paddleocr_rec_arabic_v3_line_w1024_e1 `
  --epochs 1 `
  --batch-size 1 `
  --learning-rate 0.0001 `
  --device gpu:0 `
  --num-workers 0 `
  --max-text-length 220 `
  --image-width 1024 `
  --eval-interval 500 `
  --print-interval 100
```

## OCR on Surya / CRAFT Line Annotations

Tesseract line-crop OCR:

```powershell
python scripts\ocr_surya_annotations.py `
  --images "E:\Antigravity_Code\CRAFT\surya_kurdish\images" `
  --annotations "E:\Antigravity_Code\CRAFT\surya_kurdish\annotations" `
  --tessdata-dir train\urd\output `
  --lang urd `
  --output train\urd\surya_kurdish_ocr_results.txt
```

PaddleOCR line-crop OCR:

```powershell
python scripts\ocr_surya_annotations_paddle.py `
  --images "E:\Antigravity_Code\CRAFT\surya_kurdish\images" `
  --annotations "E:\Antigravity_Code\CRAFT\surya_kurdish\annotations" `
  --output train\paddleocr_rec_arabic_v3_line_w1024_e1\surya_kurdish_ocr_results.txt `
  --device gpu:0
```

Annotation input format:

```json
{
  "file": "page_image.jpg",
  "text_lines": [
    {
      "bbox": [x, y, width, height]
    }
  ]
}
```

Line crops are sorted top-to-bottom and right-to-left within each row.

## Publication and Dataset Release Formats

Recommended public release formats:

| Format | Purpose |
|---|---|
| PAGE XML | line/region ground truth, baselines, reading order |
| ALTO XML | library OCR interoperability |
| TEI XML | scholarly text encoding |
| METS/MODS/MARC | bibliographic and archive metadata |
| JSONL | easy ML consumption |
| Parquet | large-scale tabular ML and analytics |
| PNG/TIFF/JPEG | cleaned images, preservation masters, thumbnails |
| TXT | plain raw and normalized transcription |
| CSV | metrics and compact reports |

Recommended repository release files:

- `README.ne` or `README.md`;
- `DATASET_CARD.md`;
- `DATASHEET.md`;
- `CHANGELOG.md`;
- `LICENSE`;
- `CITATION.cff`;
- schema files for metadata and annotations;
- train/validation/test split files.

## Licensing and Rights Plan

Recommended default licence for open data:

```text
CC BY-SA 4.0
```

Rights handling rules:

- KCAC must approve item selection and release status.
- Restricted items must be excluded or released only with controlled access.
- Every record should include a source attribution field naming KCAC.
- Publications and dataset users must cite KCAC and the dataset DOI.
- Commercial training of closed models should require separate KCAC consent.

## What KCAC Gets

Concrete outputs for KCAC:

1. OCR text for pilot books.
2. Better metadata in standard formats.
3. Search demonstration over the pilot.
4. Baseline OCR models and benchmark reports.
5. Co-authorship where KCAC contributes content selection, metadata, scans, or
   expert review.
6. A pipeline KCAC can keep using after the PhD work.
7. International visibility through dataset citation and publications.

## Current Findings

For the completed synthetic 10,000-image Kurdish-Sorani benchmark:

- best CER and WER: Tesseract `urd`;
- best neural OCR CER: EasyOCR `ur`;
- fastest inference: PaddleOCR PP-OCRv5;
- Farsi/Persian settings outperform Arabic settings in both Tesseract and
  EasyOCR;
- multi-language OCR settings are not automatically better;
- current custom Tesseract fine-tuning needs more work before it improves over
  base language packs.

Recommended paper wording:

```text
On the 10,000-image Kurdish-Sorani synthetic benchmark, the strongest overall
result was obtained by Tesseract with the Urdu language pack (CER 20.33%,
WER 58.75%). EasyOCR with the Urdu language list was the strongest neural OCR
configuration (CER 20.55%, WER 75.06%). PaddleOCR PP-OCRv5 produced identical
aggregate scores for ar, fa, and ur (CER 25.51%, WER 81.80%) and was the
fastest engine, averaging approximately 0.06 seconds per image on GPU.
Multi-language configurations did not improve over the best single-language
setting. Disabling Tesseract's Farsi DAWG dictionary did not change the
measured score in this run.
```

## Next Work

High-priority project tasks:

1. Confirm KCAC pilot scope, book IDs, rights status, and metadata fields.
2. Select 10-20 pilot books using a diversity matrix across era, typography,
   genre, layout, and scan quality.
3. Build a stable ID system: `kcac_NNNNNN_pNNNN_lNNNN`.
4. Convert pilot images into a clean data layout with master and cleaned
   derivatives.
5. Create PAGE XML line and region annotation workflow.
6. Produce annotation guidelines for raw transcription, normalized text,
   illegible text, punctuation, numbers, mixed-language lines, and poetry.
7. Add human verification stages and inter-annotator agreement tracking.
8. Re-run Tesseract, EasyOCR, PaddleOCR, Kraken, Calamari, and transformer OCR
   baselines on real KCAC pages.
9. Replace synthetic-only conclusions with real historical-print results.
10. Publish dataset card, datasheet, schema, splits, and baseline report.

Engineering tasks in this repository:

1. Add `requirements.txt` or `environment.yml` files for each environment.
2. Add one command runner for benchmark reproduction.
3. Move hard-coded local paths into config files.
4. Add checks that validate dataset pair format before long runs.
5. Add unit tests for normalization, CER/WER, and result merging.
6. Add README examples for all major scripts.
7. Store large generated prediction trees outside normal source control.
8. Add a clean `outputs/` or `results/` convention for new experiments.
9. Add book-level split support for future KCAC data.
10. Add export converters for PAGE XML, ALTO XML, JSONL, and Parquet.

## Meeting Topics for KCAC

Suggested agenda:

1. Confirm whether the project scope matches KCAC priorities.
2. Choose pilot books and identify sensitive or restricted material.
3. Decide scan access method: API, cloud storage, external drive, or local
   processing.
4. Clarify KCAC metadata fields and authority records.
5. Decide who can help with annotation and expert review.
6. Agree authorship and attribution rules.
7. Discuss pilot timeline, budget, and grant strategy.
8. Decide how KCAC should be named in README, dataset card, publications, and
   talks.
9. Define the first public deliverable.

## Source Notes

This README is based on:

- the project briefing document `KCAC_Briefing_Document_v2.md`;
- repository scripts under `scripts/` and `Testing/`;
- current result report `researcher/ocr_10k_results_report.md`;
- current combined benchmark output
  `Testing/new_data/results_all_models/combined_summary.txt`;
- current Tesseract training and evaluation outputs under `train/urd/`.
