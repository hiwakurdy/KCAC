activate surya:
(& conda shell.powershell hook) | Out-String | Invoke-Expression


# How To Run And Use The KCAC OCR Pipeline

This guide explains the whole process in practical order: setup, configuration, running each stage, reading outputs, and understanding why each file exists.

## 1. What This Pipeline Does

The pipeline converts local KCAC page scans into OCR ground-truth preparation files.

It runs OCR engines, aligns their line outputs, chooses consensus text, preserves raw and normalised text separately, exports PAGE XML for eScriptorium, creates a human-review queue, and writes reports/benchmarks.

The important design idea is simple: no single OCR engine is trusted alone. The pipeline compares five engines and sends uncertain lines to human review.

## 2. First-Time Setup

Run these commands from `e:\PHD\Dataset\code`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` is the Python 3.13-friendly core install. Kraken and Calamari are intentionally not in that file because the pinned OCR packages do not publish compatible Python 3.13 builds.

For your current RTX 3090 + Ollama setup, use:

```powershell
python -m pipeline --config config.ollama.example doctor
```

For the full five-engine setup with legacy Kraken/Calamari, create a Python 3.10 or 3.11 environment and then install:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-ocr-py310.txt
```

Kraken and Calamari should not be installed into the same environment with the pinned versions in this project. Kraken 5.3 requires `python-bidi~=0.6.0`, while Calamari 2.3 requires `python-bidi~=0.4.2`.

Recommended layout:

```powershell
# In surya, the main pipeline env:
python -m pip install -r requirements.txt
python -m pip install -r requirements-ocr-py310.txt

# In a separate env only for Calamari:
conda create -n calamari-ocr python=3.10
conda activate calamari-ocr
python -m pip install -r requirements-calamari-py310.txt
```

Then set the Calamari executable in `config.yaml` to the full path if it is not on `PATH`, for example:

```yaml
engines:
  calamari:
    enabled: true
    model: models/calamari/calamari_arabic_v1.ckpt
    executable: C:\Users\Hiwa\.conda\envs\calamari-ocr\Scripts\calamari-predict.exe
```

Do not activate the `calamari-ocr` environment to run `python -m pipeline` unless you also install the core pipeline requirements there. The cleaner setup is to run the pipeline from `surya` and let it call the Calamari executable by full path.

If your prompt still begins with `(.venv)` after `conda activate surya`, the project virtualenv is still active. Leave it first:

```powershell
deactivate
conda activate surya
python -c "import sys; print(sys.executable); print(sys.version)"
```

The executable should be similar to `C:\Users\Hiwa\.conda\envs\surya\python.exe`. Your `surya` env is Python 3.11, so it is a good candidate for the full OCR engine setup.

If `conda activate surya` runs but `python` still points to `C:\ProgramData\miniconda3\python.exe`, initialize the PowerShell conda hook in the current shell:

```powershell
(& conda shell.powershell hook) | Out-String | Invoke-Expression
conda activate surya
python -c "import sys; print(sys.executable); print(sys.version)"
```

For a permanent fix, run this once, close PowerShell, and open it again:

```powershell
conda init powershell
```

Confirm the command is using the venv:

```powershell
python -c "import sys; print(sys.executable)"
```

The path should start with `e:\PHD\Dataset\code\.venv\Scripts\python.exe`. If `doctor` says `No module named 'yaml'`, install requirements inside the active venv:

```powershell
python -m pip install -r requirements.txt
```

Create your working config from the example:

```powershell
Copy-Item config.yaml.example config.yaml
```

`config.yaml.example` is the safe local smoke config. It does not require Kraken OCR or Calamari checkpoints:

- Kraken is enabled with `model:` empty, so it runs line detection only.
- Calamari is disabled until a real `.ckpt` checkpoint is available.
- Qwen/Ollama and Claude are disabled in this default file.

Use `config.ollama.example` for the RTX 3090 + Ollama workflow. Use `config.full.example` only after real Kraken and Calamari model files are present.

Then edit `config.yaml`:

- Set `input.images_root` to your scan folder.
- Set model paths for Kraken and Calamari.
- Keep engines enabled only when their package, binary, model, or API key is ready.
- For your RTX 3090 + Ollama setup, keep `engines.qwen2vl.backend: ollama`.
- Set `ANTHROPIC_API_KEY` before using Claude.

Example:

```powershell
$env:ANTHROPIC_API_KEY = "your_key_here"
```

## 2.1. Use Qwen VL Through Ollama On Your RTX 3090

Because you have Ollama and an RTX 3090, the recommended Qwen setup is local Ollama inference:

```powershell
ollama list
ollama run qwen25vl-sorani-ocr:latest
```

Your `config.yaml` should contain:

```yaml
engines:
  qwen2vl:
    enabled: true
    backend: ollama
    line_source: text_lines
    model_id: qwen25vl-sorani-ocr:latest
    ollama_host: http://localhost:11434
    num_ctx: 2048
    num_predict: 256
    temperature: 0
    max_image_side: 1280
    timeout_seconds: 180
```

You can test Ollama before running the pipeline:

```powershell
curl.exe http://localhost:11434/api/chat -d "{\"model\":\"qwen25vl-sorani-ocr:latest\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}"
```

In the pipeline, Qwen does not receive the whole page. With `line_source: text_lines`, it uses only the top-level `text_lines` array from existing KCAC/Surya JSON files such as `ds_test/409/annotationa/page_0006.json`. It ignores `annotations.lines` and Kraken for this test mode. The pipeline then crops each line and sends the crop to Ollama with the transcription prompt. This is better for OCR because the VLM focuses on one line at a time and all outputs stay in the same page-pixel coordinate system.

If Ollama loads the model but the runner stops, first try lowering `num_ctx` to `1024` and `num_predict` to `64`. The model is about 19 GB on disk and can still need extra VRAM/RAM for the vision encoder and context cache.

If it still stops with an internal runner error, check:

```powershell
ollama ps
Get-Content -Tail 120 $env:LOCALAPPDATA\Ollama\server.log
```

On this machine, the installed model was detected, but Ollama 0.23.2 stopped inside the Qwen3-VL runner with a `GGML_ASSERT` from `llama-context.cpp`. That is an Ollama/model-runner issue, not a pipeline API issue. The pipeline now sends the correct local `/api/chat` payload with an image, generation options, and `stream: false`; once Ollama can answer that same request, `bootstrap` will use it.

## 3. Check The Environment

Before running OCR, use:

```powershell
python -m pipeline --config config.yaml doctor
```

`doctor` checks paths, packages, binaries, model dependencies, and API keys. If it fails, read each `FAIL` line and install or configure that dependency.

This command is intentionally strict because the plan asked for real adapters, not fake stubs.

## 4. Run One-Page Smoke Test

Use one page first to avoid wasting API budget:

```powershell
python -m pipeline --config config.yaml bootstrap --limit 42
python -m pipeline --config config.yaml consensus --limit 42
python -m pipeline --config config.yaml pagexml --limit 42
python -m pipeline --config config.yaml escriptorium --limit 42
python -m pipeline --config config.yaml queue --limit 42
python -m pipeline --config config.yaml reports --limit 42
python -m pipeline --config config.yaml hf-export --limit 42
```

After the smoke test works, run the whole configured dataset:

```powershell
python -m pipeline --config config.yaml run-all
```

Use `--force` when you want to regenerate existing outputs:

```powershell
python -m pipeline --config config.yaml bootstrap --limit 1 --force
```

## 5. Output Folders

The default output root is `output/`.

- `output/ocr_raw/{book_id}/{page_id}/{engine}.json`: raw per-engine OCR output.
- `output/consensus/{book_id}/{page_id}.json`: aligned line consensus and confidence labels.
- `output/page_xml/{book_id}/{page_id}.xml`: PAGE XML for eScriptorium.
- `output/escriptorium_import/`: side-by-side images and PAGE XML files.
- `output/annotation_queue.csv`: pages sorted for human correction.
- `output/reports/`: confidence histogram, disagreement matrix, coverage matrix, daily summary.
- `output/benchmark/`: CER, WER, and line accuracy results.
- `output/hf_dataset/lines.jsonl`: Hugging Face-ready line dataset.

## 6. File Guide And Reasoning

### Project Files

- `plans/v1.md`: Original product specification. I used it as the contract for components, outputs, and quality requirements.
- `home.html`: Existing KCAC annotation workbench. I did not rewrite it because the plan said it should remain the editor; the Python pipeline writes compatible data beside it.
- `config.yaml.example`: Safe starting configuration. I used YAML because the plan requires one central config and it is easy to edit without touching code.
- `requirements.txt`: Pinned Python dependencies. This supports reproducibility and Docker builds.
- `Dockerfile`: Reproducible runtime container. It installs Python and system OCR basics such as Tesseract.
- `pyproject.toml`: Tool settings for `ruff` and `mypy`.
- `.gitignore`: Keeps generated outputs, caches, local env files, and build artifacts out of git.

### Core Package

- `pipeline/cli.py`: Command-line entrypoint. It maps user commands like `bootstrap`, `consensus`, and `pagexml` to Python functions.
- `pipeline/config.py`: Loads `config.yaml`. This keeps paths, engine settings, thresholds, and report options outside the code.
- `pipeline/models.py`: Shared typed data models. I used dataclasses so JSON output stays clear while Python code gets type safety.
- `pipeline/discovery.py`: Finds images and creates stable `book_id` and `page_id` values.
- `pipeline/jsonio.py`: Central JSON read/write helpers, so UTF-8 and formatting are consistent.
- `pipeline/geometry.py`: Polygon area, IoU, boxes, and PAGE point strings. This is needed for line alignment and XML export.
- `pipeline/retry.py`: Retry and consecutive-failure guard. This protects budget and stops after repeated API failures.
- `pipeline/budget.py`: Tracks estimated API spend events.

### OCR And Consensus

- `pipeline/bootstrap.py`: Runs enabled engines sequentially and writes raw OCR JSON. Sequential execution was chosen to control API cost and rate limits.
- `pipeline/engines/tesseract.py`: Tesseract adapter using `pytesseract` with `ckb`. It is the local baseline.
- `pipeline/engines/kraken.py`: Kraken adapter and line detector. Kraken is important because the VLMs use its line crops instead of doing layout themselves.
- `pipeline/engines/calamari.py`: Calamari adapter on Kraken line crops. It is included for historical print and letterpress-style OCR.
- `pipeline/engines/qwen2vl.py`: Qwen VL adapter. It supports Ollama for your local RTX 3090 workflow and Transformers as a fallback backend.
- `pipeline/engines/claude.py`: Claude adapter through the Anthropic SDK. It is used for difficult line crops when API budget allows.
- `pipeline/consensus.py`: Aligns engine lines by polygon IoU, computes edit distance, votes on text, and labels each line as `auto_accept`, `near_agreement`, or `disagreement`.
- `pipeline/normalise.py`: Sorani Unicode normaliser. It preserves raw text and writes normalised text plus a trace log.

### Exports And Review

- `pipeline/pagexml_export.py`: Writes PAGE XML 2019-07-15. This is the main interchange format for eScriptorium.
- `pipeline/escriptorium_import.py`: Creates the image/XML sidecar layout eScriptorium can import.
- `pipeline/queue.py`: Builds the human review queue, prioritising disagreement and dense pages.
- `pipeline/reports.py`: Writes project summaries and agreement reports after runs.
- `pipeline/benchmark.py`: Computes CER, WER, and line-level accuracy after corrected ground truth exists.
- `pipeline/hf_export.py`: Writes line-level JSONL for Hugging Face `datasets`.

### Documentation And Tracking

- `docs/normalisation_policy.md`: Explains every Unicode mapping and why raw text is never overwritten.
- `docs/page_xml_schema_compliance.md`: Explains how PAGE XML is structured and validated.
- `docs/tracking/linear_backlog.md`: Local Linear backlog plan. It exists because external Linear creation was intentionally deferred.
- `docs/tracking/github_issues.md`: Local GitHub issue plan for milestone `v0.1`.
- `.github/ISSUE_TEMPLATE/`: Ready-to-use GitHub issue templates once a remote repository is connected.

## 7. Why These Models And Engines

- Tesseract `ckb`: Cheap, local baseline for Sorani OCR.
- Kraken: Strong historical/Arabic-script OCR ecosystem and useful line segmentation.
- Calamari: Useful for historical typefaces and line-level OCR experiments.
- Qwen VL through Ollama: Open local VLM path for hard line crops, using your RTX 3090 through the Ollama server.
- Claude: API VLM fallback for the hardest pages, controlled by rate limits and budget settings.

The VLMs are asked to transcribe line crops only. They are not asked to perform layout and transcription together because that usually lowers consistency. Kraken supplies the line geometry so every engine result stays in the same page-pixel coordinate system.

## 8. Human Correction Process

1. Run OCR bootstrap.
2. Build consensus.
3. Export PAGE XML.
4. Build eScriptorium import folder.
5. Import `output/escriptorium_import/` into eScriptorium.
6. Correct lines with `disagreement` and `near_agreement` first.
7. Re-export corrected PAGE XML.
8. Run benchmark and Hugging Face export.

The `annotation_queue.csv` tells annotators which pages to correct first.

## 9. Testing And Quality Checks

Run:

```powershell
python -m pytest
python -m ruff check .
python -m mypy pipeline
```

Tests cover Unicode normalisation, polygon IoU, consensus voting, retry/budget logic, PAGE XML structure, and compatibility with the existing `ds_test/409` KCAC JSON sample.

## 10. Practical Notes

- Start with `--limit 1`.
- Keep Claude disabled until the API key and budget are ready.
- Keep Qwen enabled through Ollama on your RTX 3090; disable it only if Ollama is not running or the model is not downloaded.
- Do not edit raw OCR JSON by hand; correct text in eScriptorium or the KCAC workbench.
- Use `--force` only when you want to overwrite generated stage outputs.
