from __future__ import annotations

import importlib.util
import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


@dataclass(slots=True)
class CheckResult:
    name: str
    ok: bool
    message: str


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _ollama_models(host: str) -> tuple[bool, list[str], str]:
    endpoint = f"{host.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(endpoint, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, [], str(exc)
    models = []
    for item in data.get("models", []):
        if isinstance(item, dict) and item.get("name"):
            models.append(str(item["name"]))
    return True, models, endpoint


def _ollama_has_model(models: list[str], model: str) -> bool:
    return model in models or f"{model}:latest" in models or any(item.split(":", 1)[0] == model for item in models)


def _command_available(command: str) -> bool:
    path = Path(command)
    if path.is_absolute() or "\\" in command or "/" in command:
        return path.exists()
    return shutil.which(command) is not None


def _configured_model_exists(config_path: Path, model: str | None) -> tuple[bool, str]:
    if not model:
        return False, "model path is not configured"
    path = Path(model)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path.exists(), str(path)


def run_doctor(config: PipelineConfig) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.append(CheckResult("images_root", config.images_root.exists(), str(config.images_root)))
    results.append(CheckResult("output_root_parent", config.output_root.parent.exists(), str(config.output_root.parent)))
    results.append(CheckResult("pillow", _module_available("PIL"), "Python pillow package"))

    if config.engines.get("tesseract", None) and config.engines["tesseract"].enabled:
        results.append(CheckResult("pytesseract", _module_available("pytesseract"), "Python pytesseract package"))
        results.append(CheckResult("tesseract_binary", shutil.which("tesseract") is not None, "tesseract executable on PATH"))
    if config.engines.get("kraken", None) and config.engines["kraken"].enabled:
        results.append(CheckResult("kraken", _module_available("kraken"), "Python kraken package"))
        if config.engines["kraken"].model:
            ok, message = _configured_model_exists(config.config_path, config.engines["kraken"].model)
            results.append(CheckResult("kraken_model", ok, message))
        else:
            results.append(CheckResult("kraken_model", True, "not configured; Kraken will run line detection only"))
    if config.engines.get("calamari", None) and config.engines["calamari"].enabled:
        executable = str(config.engines["calamari"].extra.get("executable", "calamari-predict"))
        results.append(CheckResult("calamari-predict", _command_available(executable), executable))
        ok, message = _configured_model_exists(config.config_path, config.engines["calamari"].model)
        results.append(CheckResult("calamari_model", ok, message))
    if config.engines.get("qwen2vl", None) and config.engines["qwen2vl"].enabled:
        qwen_cfg = config.engines["qwen2vl"]
        backend = str(qwen_cfg.extra.get("backend", "transformers")).lower()
        if backend == "ollama":
            host = str(qwen_cfg.extra.get("ollama_host", DEFAULT_OLLAMA_HOST))
            ok, models, message = _ollama_models(host)
            model = qwen_cfg.model_id or qwen_cfg.model or "qwen25vl-sorani-ocr:latest"
            results.append(CheckResult("ollama_server", ok, message))
            results.append(CheckResult("ollama_model", _ollama_has_model(models, model), f"{model} in Ollama model list"))
        else:
            results.append(CheckResult("transformers", _module_available("transformers"), "Python transformers package"))
            results.append(CheckResult("torch", _module_available("torch"), "Python torch package"))
    if config.engines.get("claude", None) and config.engines["claude"].enabled:
        results.append(CheckResult("anthropic", _module_available("anthropic"), "Python anthropic package"))
        results.append(CheckResult("ANTHROPIC_API_KEY", bool(os.getenv("ANTHROPIC_API_KEY")), "Anthropic API key"))
    return results


def format_doctor(results: list[CheckResult]) -> str:
    lines = []
    for result in results:
        status = "OK" if result.ok else "FAIL"
        lines.append(f"{status:4} {result.name}: {result.message}")
    return "\n".join(lines)
