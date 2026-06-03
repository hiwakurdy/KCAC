# Critical Review: KCAC OCR Pipeline v0.1

> **Reviewer**: External AI/ML research critique
> **Subject**: KCAC OCR Pipeline (5-engine consensus, PAGE XML, Sorani normalisation)
> **Reviewer's posture**: Honest, prioritised, actionable. No padding, no flattery.
> **Response plan**: See `docs/plans/Pipeline_Critical_Review_Response.md` for answers, decisions, schedule, and implementation ownership.

---

## Summary

The pipeline is architecturally strong: five-engine consensus, sequential cost control, PAGE XML schema compliance, trace-logged Unicode normalisation, Hugging Face JSONL export, eScriptorium-compatible review folders, and pytest/ruff/mypy quality gates are all present. This is **professional infrastructure**, not a PhD-script-dump.

However, **shipping v0.1 today would be premature**. The pipeline produces *outputs*; it does not yet produce *evidence those outputs are research-grade*. The gaps below cluster into three priority tiers.

---

## PRIORITY 1 — Must fix before v0.1 public release

### P1.1 — Sorani normalisation is incomplete

The current policy handles 4 mappings (Yeh, Alef Maksura, Kaf, tashkeel). For historical Sorani print, **at least 6 more are required**, and silent omissions will be cited against the dataset by any Kurdish linguist reviewer:

| Missing mapping | Problem | Fix |
|---|---|---|
| Heh forms: ه (U+0647) vs ة (U+0629) vs ە (Kurdish E, U+06D5) | Historically conflated; ە is the Sorani E sound, NOT an Arabic letter | Add explicit rule, document carefully — ə is **not** always normalisable to ه |
| Waw variants: و / ۆ / وو / ۊ | Different vowels in Kurdish; old prints often use و where modern prints use ۆ or وو | Decision required: keep as-printed, or apply contextual mapping. Recommend: **keep as-printed in raw layer**, do NOT touch in normalised layer |
| Hamza-bearing letters: أ ؤ ئ إ ء | Different Unicode codepoints depending on bearer | Apply NFC + document policy on bearer choice |
| Lam-Alef ligatures: ﻻ (U+FEFB), ﻷ etc. | OCR may emit ligature codepoints; downstream tools want decomposed | Decompose ligatures in normalised layer |
| ZWNJ (U+200C) | Sorani compound words use ZWNJ; old typewriter prints omit it | Decision required: do NOT silently insert; flag as `normalisation_note` |
| Final/medial form differences | OCR may emit presentation forms (FE..) instead of base codepoints | Normalise to base forms in normalised layer |

**Action**: Expand `docs/normalisation_policy.md` to cover all 10 cases with: codepoint, name, mapping decision, **and rationale signed off by a Kurdish linguist**. Add unit tests with explicit before/after pairs for every documented mapping.

### P1.2 — No baseline comparison against prior work

You mentioned previous experiments with Tesseract, PaddleOCR, and EasyOCR on Kurdish. **The current pipeline does not yet demonstrate it is better than your own prior baselines.** Without that comparison, a reviewer will reasonably ask: "Why is 5-engine consensus worth the engineering cost over just using the best single engine?"

**Action**: Before public release, produce `benchmark/baseline_comparison.md` with:

- CER and WER for: Tesseract alone, PaddleOCR alone, EasyOCR alone, Qwen2.5-VL alone, **consensus**.
- Per-bucket comparison (era, typography, layout density).
- Cost comparison: total run time and total API cost per book.
- Honest conclusion: where does consensus beat single-engine, and where does it not?

If consensus only improves the hardest 10% of lines, that's fine — but report it honestly.

### P1.3 — Pre-OCR image preprocessing is missing

For historical scans, **image preprocessing typically buys more CER reduction than choosing a fancier model**. The current pipeline jumps straight to OCR engines. Missing stages:

1. **Deskew** — historical scans are rarely perfectly aligned; even 1° skew kills line-detection accuracy
2. **Denoise** — speckle and bleed-through from facing pages
3. **Contrast normalisation** — old prints often have low contrast or yellow-aged paper
4. **Binarisation (Sauvola / Otsu / adaptive)** — some engines (Tesseract especially) prefer bitonal input; others (VLMs) prefer original
5. **Resolution check** — flag pages below 200 DPI estimate; OCR quality collapses below ~150 DPI

**Action**: Add `pipeline/preprocess.py` with the five stages above, configurable per engine. Tesseract should receive Sauvola-binarised input; VLMs should receive original colour. Store preprocessed derivatives in `output/preprocessed/{book_id}/{page_id}/`.

### P1.4 — Annotator workflow is undefined

The pipeline outputs eScriptorium-compatible files and an annotation queue, but the **human side of the loop is undocumented**:

- Who is the annotator? (Recruitment criteria, Kurdish proficiency level required)
- What does "done" look like per page? (Definition of done)
- How are two annotators reconciled when they disagree?
- How is **inter-annotator agreement (IAA)** measured? (Cohen's kappa, char-F1)
- What's the per-page review-time budget?
- What's the training material for new annotators?

Without this, the human pipeline is non-reproducible. The senior linguist reviewer (P1.1) is the gatekeeper but **the front-line annotators need explicit guidance**.

**Action**: Write `docs/annotator_handbook.md` covering:
- Recruitment: 2 Sorani native speakers, university-level literacy, 10–20 hr/week available
- Per-page protocol: "First verify line boxes, then correct text, then verify metadata flags"
- Reconciliation: senior linguist adjudicates after second-pass review
- IAA: every 100th page, computed automatically and surfaced in `output/reports/iaa.csv`
- Training: 10 pre-annotated example pages with rationale notes, used as onboarding

---

## PRIORITY 2 — Important for quality, fix before paper submission

### P2.1 — Line-detection quality not measured

If Kraken (or any engine) misses 5% of lines or merges adjacent lines, the transcription is *wasted on missing or corrupted geometry*. The current pipeline measures text quality but not line-detection quality.

**Action**: Add `pipeline/line_eval.py` with:
- Per-page line-recall and line-precision against human-corrected ground truth (once available)
- Polygon IoU distribution
- Flag pages where line count differs from human-detected by > 10%

### P2.2 — Confidence calibration not done

`auto_accept` is defined as "3+ engines produce identical text". But **is that actually a good signal?** Without calibration, we don't know:

- Of lines marked `auto_accept`, what % are actually correct?
- Of lines marked `disagreement`, what % could have been resolved automatically?

It's plausible that 3 weak engines often agree on the same wrong reading (Tesseract, Kraken, Calamari may share blind spots).

**Action**: After human correction is available for a sample, compute and publish:
- Calibration curve: predicted confidence vs actual accuracy
- Optimal threshold per page-type bucket
- ROC of the auto-accept decision

### P2.3 — Pipeline stress testing absent

Has the pipeline run end-to-end on **a full 80-page book** with all engines enabled? Documented behaviours that may need verification:

- Memory leaks on long runs
- Ollama process recovery if it crashes mid-book
- API budget alerts firing correctly
- Resumability after force-quit
- Disk-space behaviour with 5 engines × 80 pages = 400 JSON files per book

**Action**: Run an unattended overnight test on at least one full book. Capture: total wall time, peak RAM, peak GPU memory, total disk written, API cost, any non-zero exit codes from sub-processes.

### P2.4 — Active learning loop is open

The pipeline produces a `annotation_queue.csv` but **once humans correct, nothing flows back into the pipeline**. Specifically:

- Corrected lines are not used to retrain Kraken line detection
- Corrected text is not used to retrain a custom OCR model on Sorani fonts
- Consensus thresholds are not adapted to observed accuracy
- Common annotator-corrections are not surfaced as recurring engine error patterns

**Action**: Add a closing-loop stage `pipeline/feedback.py` that:
- After every 100 corrected pages, retrains a Kraken line model on corrected geometry
- Logs the top 20 most-corrected engine-output patterns (helps identify systemic engine bugs)
- Recommends consensus-threshold adjustments based on observed CER per bucket

This is the difference between a static dataset and a *living* annotation pipeline.

### P2.5 — Edge-case page behaviour unclear

The PAGE XML schema accommodates many region types, but the v0.1 pipeline puts everything in one `TextRegion`. What happens when the page is:

- **Multi-column** (periodical, dictionary) — reading order will be wrong
- **Poetry centred** — line geometry is unusual
- **A table** — current schema does not capture table structure
- **Handwritten margins** — engines may transcribe them as if printed
- **Mixed Arabic + Kurdish** — language ID per region is missing
- **Predominantly an illustration with caption** — caption may be missed

**Action**: For v0.1, **explicitly exclude** edge-case pages from the dataset (filter at discovery stage) and document the exclusion. For v0.5, add proper region typing. Honest scoping beats silent failures.

---

## PRIORITY 3 — Polish for publication

### P3.1 — Datasheet for Datasets not yet written

For the published release on Zenodo / Hugging Face, you need a **Datasheet for Datasets** (Gebru et al. 2018) in the form expected by NeurIPS / ICDAR / LREC reviewers. The README is not a substitute.

**Action**: Use the standard template at <https://arxiv.org/abs/1803.09010>. Cover: motivation, composition, collection process, preprocessing, uses, distribution, maintenance. ~6–8 pages.

### P3.2 — Architecture diagram is missing

Reviewers and collaborators need a visual flow of: scans → preprocessing → 5 engines → consensus → PAGE XML → review → corrections → benchmark.

**Action**: Add `docs/architecture.md` with a Mermaid diagram (renders on GitHub). One page.

### P3.3 — Demo notebook absent

A Jupyter notebook that loads the published dataset, displays a sample page with line boxes, and computes a quick CER on a baseline model would dramatically lower the adoption barrier for downstream researchers.

**Action**: Add `notebooks/quickstart.ipynb`. Should run on Colab with the published dataset in under 5 minutes.

### P3.4 — Train/val/test split not formalised for current data

The pipeline supports book-level splitting via `config.yaml`, but the **actual book selection** is not yet documented. Reviewers will want to see:

- Which exact KCAC book IDs are in train / val / test
- Justification for each (era, script, typography coverage)
- A coverage matrix showing the test set spans the diversity axes

**Action**: Write `splits/SPLITS.md` with the final book lists, rationale, and a per-bucket coverage matrix.

### P3.5 — Per-engine model versioning

The pipeline uses Tesseract `ckb`, Kraken model X, Calamari model Y, Qwen `qwen2.5vl:7b-fp16`, Claude `claude-sonnet-4-7`. **Each of these will drift over time**. Without locking versions, the benchmark is not reproducible 6 months from now.

**Action**: Add `pipeline/version_lock.py` that on every run captures: Tesseract version, Kraken model SHA-256, Calamari model SHA-256, Ollama model digest, Claude model ID. Write to `output/run_manifest.json`.

---

## What is ALREADY GOOD (so this critique is fair)

For the avoidance of doubt:

- ✓ Five-engine consensus design is novel for Kurdish OCR and well-justified
- ✓ Sequential execution with budget tracking is mature and respectful
- ✓ PAGE XML schema compliance is a strong choice
- ✓ Dual `TextEquiv` (raw + normalised) avoids the single-most-common pitfall in heritage OCR datasets
- ✓ Trace-logged Unicode normalisation is rare and impressive
- ✓ ruff + mypy + pytest gates show engineering discipline
- ✓ Sequential rather than parallel API calls is the right cost choice for v0.1
- ✓ Separation of `bootstrap → consensus → pagexml → escriptorium → queue → reports → benchmark → hf-export` is clean and resumable
- ✓ Documentation of why each file exists (in `how_to_run_and_use.md`) is unusually good

This pipeline is **infrastructure**. The above gaps are not signs of weakness — they are signs of *what to do next*, in a project that has cleared the foundational engineering bar.

---

## Suggested Sequence

Do them in this order (assumes scans for v0.1 books are now available):

1. **Days 1–3**: Fix P1.1 (Sorani normalisation) + P1.3 (image preprocessing). Re-run bootstrap on one book.
2. **Days 4–5**: Run the P1.2 baseline comparison. Honest report.
3. **Days 6–8**: Write P1.4 (annotator handbook). Onboard 2 annotators on 10 pre-annotated example pages.
4. **Days 9–16**: Annotators work. Pipeline lead runs P2.3 (stress test) on a second book in parallel.
5. **Days 17–20**: P2.1 (line eval) + P2.2 (calibration) computed on first corrected book.
6. **Days 21–24**: P2.4 (feedback loop) + P2.5 (edge case filter) implemented.
7. **Days 25–28**: P3.1 (datasheet) + P3.2 (architecture diagram) + P3.3 (notebook) + P3.4 (splits doc) + P3.5 (version lock).
8. **Days 29–30**: Final review, dataset versioning, Zenodo deposit, Hugging Face upload, paper draft.

This is the realistic path to a release that survives peer review and that other researchers actually use.

---

## One Final Honest Thought

The biggest unknown right now is **P1.2 — the baseline comparison**. If 5-engine consensus only beats single Tesseract by 1–2 CER points, the engineering cost is not justified and the v0.1 paper has no story. If it beats by 5+ CER points, you have a top-tier ICDAR paper. **Run that comparison early. It tells you whether the rest of the work is shaped correctly.**

Everything else is iterating on a good plan. P1.2 might tell you the plan needs to shift.
