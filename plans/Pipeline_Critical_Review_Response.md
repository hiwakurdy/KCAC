# Response Plan: KCAC OCR Pipeline Critical Review

> Source review: `plans/Pipeline_Critical_Review.md`  
> Purpose: Answer every question raised in the review and define what will be done, when, how, and where in the repository.  
> Schedule assumption: Day 1 starts on 2026-05-12.

---

## Executive Answer

We will not publish v0.1 immediately. The codebase can produce OCR pipeline artifacts, but the review is correct: publication needs stronger evidence, preprocessing, annotator protocol, baseline comparison, calibration, stress testing, version locking, and release documentation.

The immediate path is:

1. Fix Sorani normalisation and preprocessing first.
2. Run an early baseline comparison to prove whether five-engine consensus is worth it.
3. Document the human annotation protocol before annotators start.
4. Add evaluation and feedback stages after corrected pages exist.
5. Add publication artifacts only after the core evidence exists.

---

## Timeline

| Dates | Days | Work |
|---|---:|---|
| 2026-05-12 to 2026-05-14 | 1-3 | P1.1 normalisation + P1.3 preprocessing |
| 2026-05-15 to 2026-05-16 | 4-5 | P1.2 baseline comparison |
| 2026-05-17 to 2026-05-19 | 6-8 | P1.4 annotator handbook + onboarding package |
| 2026-05-20 to 2026-05-27 | 9-16 | Annotation work + P2.3 stress test |
| 2026-05-28 to 2026-05-31 | 17-20 | P2.1 line eval + P2.2 confidence calibration |
| 2026-06-01 to 2026-06-04 | 21-24 | P2.4 feedback loop + P2.5 edge-case filtering |
| 2026-06-05 to 2026-06-08 | 25-28 | P3 publication docs, notebook, splits, version lock |
| 2026-06-09 to 2026-06-10 | 29-30 | Final review, dataset versioning, release prep |

---

## P1.1 Sorani Normalisation

### Questions Answered

**Should Heh forms be normalised automatically?**  
No, not blindly. U+0647 Arabic Heh, U+0629 Teh Marbuta, and U+06D5 Kurdish E must be documented separately. U+06D5 may be normalised only when the printed/OCR character clearly represents Sorani E. Otherwise the raw layer remains authoritative and the line receives a normalisation note.

**Should Waw variants be mapped?**  
No automatic contextual mapping in v0.1. Waw variants can represent different Kurdish vowels and historical spelling conventions. We preserve them in raw text and keep the normalised layer conservative unless a linguist-approved rule exists.

**How do we handle hamza-bearing letters?**  
Apply Unicode NFC and document bearer policy. Do not silently change the intended bearer unless it is a Unicode presentation/compatibility issue.

**How do we handle Lam-Alef ligatures?**  
Decompose presentation ligatures in the normalised layer. Keep raw OCR unchanged.

**Should ZWNJ be inserted?**  
No. We never silently insert ZWNJ. Missing ZWNJ is recorded as a `normalisation_note` for human review.

**How do we handle Arabic presentation forms?**  
Normalised text converts presentation forms to base Unicode codepoints using NFKC/NFC policy, while raw text remains untouched.

### What We Will Do

- Expand `pipeline/normalise.py` with documented rules for:
  - Arabic/Persian Yeh
  - Alef Maksura
  - Arabic/Kurdish Kaf
  - Heh/Kurdish E policy
  - Waw non-mapping policy
  - Hamza-bearing letters
  - Lam-Alef ligatures
  - ZWNJ notes
  - Arabic presentation forms
  - Tashkeel stripping
- Expand `docs/normalisation_policy.md` into a rule table with codepoint, Unicode name, mapping decision, rationale, and linguist sign-off status.
- Add tests in `tests/test_normalise.py`.

### Acceptance Criteria

- Every documented rule has at least one explicit before/after unit test.
- Raw text is never modified.
- Normalisation trace includes the reason for every changed codepoint.
- Rules requiring expert approval are marked `pending_linguist_signoff` until signed.

---

## P1.2 Baseline Comparison

### Questions Answered

**Why is five-engine consensus worth the engineering cost?**  
We do not claim that until measured. The comparison will decide whether consensus is justified. If consensus improves only a small subset, we will report that honestly and use consensus mainly as a hard-line triage tool.

**Which baselines are compared?**  
Tesseract, PaddleOCR, EasyOCR, Qwen via Ollama, and consensus.

### What We Will Do

- Add `pipeline/baseline_compare.py`.
- Produce `output/benchmark/baseline_comparison.md`.
- Compare:
  - CER
  - WER
  - Line exact accuracy
  - Per-bucket scores by era, typography, and layout density
  - Runtime per book
  - API or compute cost per book

### How

- Use corrected PAGE XML/KCAC JSON as gold.
- Read engine outputs from `output/ocr_raw/`.
- Add import adapters for prior PaddleOCR/EasyOCR results if stored externally.
- Reuse existing `pipeline/benchmark.py` edit-distance utilities.

### Acceptance Criteria

- Report includes a table for each engine and consensus.
- Report states where consensus wins and where it fails.
- Report includes at least one hard-page subset analysis.

---

## P1.3 Pre-OCR Image Preprocessing

### Questions Answered

**Which preprocessing stages are needed?**  
Deskew, denoise, contrast normalisation, binarisation, and resolution checks.

**Do all engines receive the same image?**  
No. Tesseract gets a binarised derivative. Kraken/Calamari can use binarised or contrast-normalised derivatives depending on config. VLMs receive original colour or lightly resized colour crops.

### What We Will Do

- Add `pipeline/preprocess.py`.
- Add CLI command `preprocess`.
- Write derivatives to `output/preprocessed/{book_id}/{page_id}/`.
- Add config section:
  - `preprocess.enabled`
  - `deskew`
  - `denoise`
  - `contrast`
  - `binarisation`
  - `min_dpi_warning`
  - per-engine derivative choice

### How

- Use Pillow and optional scikit-image/OpenCV when available.
- Implement Otsu first because it is dependency-light.
- Add Sauvola when scikit-image is installed.
- Store a JSON sidecar with preprocessing parameters and warnings.

### Acceptance Criteria

- Tesseract adapter can read preprocessed page path.
- VLM adapters keep using original colour crops.
- Low-resolution pages are flagged in reports.

---

## P1.4 Annotator Workflow

### Questions Answered

**Who is the annotator?**  
Two Sorani native speakers with university-level literacy, historical print familiarity preferred, available 10-20 hours per week.

**What does done mean per page?**  
A page is done when line boxes are checked, reading order is correct, raw text is corrected, normalised text is reviewed, metadata flags are filled, and QC passes.

**How are disagreements reconciled?**  
First and second annotator corrections are compared. A senior linguist adjudicates disagreements and signs off gold pages.

**How is IAA measured?**  
Every 100th page receives double annotation. IAA is computed as char-F1, line exact agreement, and Cohen-style categorical agreement for metadata/QC flags.

**What is the per-page time budget?**  
Target: 6-10 minutes for clean pages, 12-20 minutes for hard pages. Pages exceeding 20 minutes are flagged for senior review.

**What is the training material?**  
Ten pre-annotated example pages with rationale notes: clean body text, low contrast, poetry, table-like page, title page, dense page, damaged scan, mixed orthography, marginalia, and hard historical typography.

### What We Will Do

- Add `docs/annotator_handbook.md`.
- Add `pipeline/iaa.py`.
- Add `output/reports/iaa.csv` generation.

### Acceptance Criteria

- Handbook has a definition of done.
- Handbook has examples of raw vs normalised text.
- IAA report can be generated from two corrected annotation folders.

---

## P2.1 Line-Detection Evaluation

### Question Answered

**How do we know OCR text is not wasted on bad geometry?**  
We measure line recall, line precision, line count mismatch, and polygon IoU against human-corrected geometry.

### What We Will Do

- Add `pipeline/line_eval.py`.
- Output:
  - `output/reports/line_eval.csv`
  - `output/reports/line_iou_histogram.png`

### Acceptance Criteria

- Pages with >10% line-count mismatch are flagged.
- Mean/median IoU is reported per book and engine.

---

## P2.2 Confidence Calibration

### Questions Answered

**Are auto-accepted lines actually correct?**  
We will measure this after corrected pages exist. Until then, `auto_accept` is a heuristic label, not a proven probability.

**Can disagreements be resolved automatically?**  
Only after calibration. The calibration report will identify safe thresholds.

### What We Will Do

- Add `pipeline/calibration.py`.
- Output:
  - `output/reports/confidence_calibration.csv`
  - `output/reports/confidence_calibration.png`
  - ROC data for auto-accept decisions

### Acceptance Criteria

- Actual accuracy is reported for every confidence label.
- Threshold recommendations are bucketed by page type/layout.

---

## P2.3 Stress Testing

### Questions Answered

**Has the full system run on an 80-page book?**  
Not yet. This is required before paper submission.

**What will be measured?**  
Wall time, peak RAM, peak GPU memory, disk written, API spend, subprocess failures, resume behavior, and Ollama recovery.

### What We Will Do

- Add `pipeline/stress.py`.
- Add `docs/stress_test_report_template.md`.
- Run at least one 80-page unattended book test.

### Acceptance Criteria

- Stress report includes command, config, machine specs, peak resources, failures, and rerun/resume result.

---

## P2.4 Feedback Loop

### Questions Answered

**How do corrected pages improve the pipeline?**  
Every 100 corrected pages trigger feedback analysis. Model retraining is recommended or launched depending on config.

**What is learned?**  
Common engine mistakes, threshold adjustment recommendations, and Kraken retraining candidates.

### What We Will Do

- Add `pipeline/feedback.py`.
- Output:
  - `output/reports/top_engine_corrections.csv`
  - `output/reports/threshold_recommendations.md`
  - optional Kraken retraining command script

### Acceptance Criteria

- Top 20 correction patterns are reported.
- Threshold recommendations include evidence, not guesses.

---

## P2.5 Edge-Case Page Behavior

### Questions Answered

**What happens to multi-column, poetry, tables, handwritten margins, mixed language, and illustration-heavy pages in v0.1?**  
They are excluded from the public v0.1 training/evaluation set unless explicitly marked as accepted v0.1 scope. They can stay in a separate `excluded` manifest for future v0.5 work.

### What We Will Do

- Add `pipeline/filters.py`.
- Add config section `scope.exclude_page_types`.
- Write `output/exclusions/edge_cases.csv`.
- Document exclusions in `docs/scope_v0_1.md`.

### Acceptance Criteria

- Excluded pages are listed with reason.
- Public dataset card states v0.1 excludes complex layout cases.

---

## P3.1 Datasheet For Datasets

### What We Will Do

- Add `docs/datasheet_for_datasets.md`.
- Use Gebru et al. sections:
  - Motivation
  - Composition
  - Collection process
  - Preprocessing
  - Uses
  - Distribution
  - Maintenance

### Acceptance Criteria

- Datasheet is complete before Zenodo/Hugging Face release.
- Licensing and KCAC attribution are explicit.

---

## P3.2 Architecture Diagram

### What We Will Do

- Add `docs/architecture.md`.
- Include Mermaid flow:
  - scans
  - preprocessing
  - five OCR engines
  - consensus
  - PAGE XML
  - eScriptorium review
  - corrected gold
  - benchmark
  - release

### Acceptance Criteria

- Mermaid renders on GitHub.
- Diagram matches actual CLI stages.

---

## P3.3 Demo Notebook

### What We Will Do

- Add `notebooks/quickstart.ipynb`.
- Notebook will:
  - load a sample dataset row
  - display page image
  - draw line polygons
  - compute quick CER
  - show PAGE XML/HF JSONL fields

### Acceptance Criteria

- Runs in Colab in under 5 minutes with published sample data.

---

## P3.4 Train/Val/Test Split Documentation

### What We Will Do

- Add `splits/SPLITS.md`.
- Include exact book IDs for train, validation, and test.
- Include rationale by era, script, typography, layout density, and scan quality.
- Include coverage matrix.

### Acceptance Criteria

- No random page-level split.
- Test set contains whole books only.

---

## P3.5 Per-Engine Version Lock

### Questions Answered

**How do we reproduce benchmarks later?**  
Every run writes a manifest with executable versions, model IDs, model hashes, and Ollama digests.

### What We Will Do

- Add `pipeline/version_lock.py`.
- Write `output/run_manifest.json`.
- Capture:
  - Git commit if available
  - Python version
  - OS
  - Tesseract version
  - Kraken package version
  - Kraken model SHA-256
  - Calamari executable path/version
  - Calamari model SHA-256
  - Ollama model digest
  - Claude model ID
  - config SHA-256

### Acceptance Criteria

- `run_manifest.json` is created at the start of `run-all`.
- Benchmark reports reference the manifest path.

---

## Immediate Commands

### Ollama/Qwen-Only Path

Use this when working in the current Python 3.13 `.venv` or when you want fast local Qwen testing:

```powershell
python -m pip install -r requirements.txt
python -m pipeline --config config.ollama.example doctor
python -m pipeline --config config.ollama.example bootstrap --limit 1 --force
```

### Full Main Pipeline Env

Use this in the `surya` Python 3.11 environment:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-ocr-py310.txt
python -m pipeline --config config.yaml.example doctor
```

### Separate Calamari Env

Use this only if Calamari is required:

```powershell
conda create -n calamari-ocr python=3.10
conda activate calamari-ocr
python -m pip install -r requirements-calamari-py310.txt
```

Then point `config.yaml` at the Calamari executable:

```yaml
engines:
  calamari:
    enabled: true
    model: calamari_arabic_v1.ckpt
    executable: C:\Users\Hiwa\.conda\envs\calamari-ocr\Scripts\calamari-predict.exe
```

---

## Release Gate

v0.1 can be public only when all these are true:

- P1.1 normalisation table is complete and reviewed.
- P1.2 baseline comparison exists.
- P1.3 preprocessing is implemented and documented.
- P1.4 annotator handbook exists.
- At least one corrected sample has line eval and calibration reports.
- `output/run_manifest.json` exists for the release run.
- Splits are book-level and documented.
- Datasheet and architecture docs are present.

If any item is missing, the dataset can still be used internally, but not claimed as research-grade public v0.1.
