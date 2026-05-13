from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from ..line_sources import load_kcac_json_lines
from ..models import BudgetEvent, EnginePageOutput, LineOutput, PageRef
from .base import BaseOcrEngine, EngineError, crop_line, image_to_base64_jpeg, make_output
from .kraken import KrakenEngine

DEFAULT_TRANSFORMERS_MODEL = "Qwen/Qwen2-VL-72B-Instruct"
DEFAULT_OLLAMA_MODEL = "qwen25vl-sorani-ocr:latest"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
LINE_TRANSCRIPTION_PROMPT = (
    "Transcribe this single Kurdish Sorani Arabic-script print line exactly as printed. "
    "Return only the transcription, with no explanation."
)


class Qwen2VlEngine(BaseOcrEngine):
    name = "qwen2vl"

    @property
    def backend(self) -> str:
        return str(self.config.extra.get("backend", "transformers")).lower()

    def version(self) -> str:
        if self.backend == "ollama":
            model_id = self.config.model_id or self.config.model or DEFAULT_OLLAMA_MODEL
            host = str(self.config.extra.get("ollama_host", DEFAULT_OLLAMA_HOST)).rstrip("/")
            return f"{model_id} via Ollama at {host}"
        transformers = self.require_import("transformers")
        model_id = self.config.model_id or DEFAULT_TRANSFORMERS_MODEL
        return f"{model_id} via transformers {getattr(transformers, '__version__', '')}"

    def run(self, page: PageRef) -> EnginePageOutput:
        if self.backend == "ollama":
            return self._run_ollama(page)
        if self.backend != "transformers":
            raise EngineError(self.name, f"Unsupported qwen2vl backend: {self.backend}", recoverable=False)
        return self._run_transformers(page)

    def _run_transformers(self, page: PageRef) -> EnginePageOutput:
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise EngineError(self.name, "transformers, torch, and Qwen2VL support are required", recoverable=False) from exc

        model_id = self.config.model_id or DEFAULT_TRANSFORMERS_MODEL
        device = "cuda" if self.config.use_local_gpu and torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_id)  # type: ignore[no-untyped-call]
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model.to(device)

        detected = self._line_boxes(page)
        lines: list[LineOutput] = []
        for line in detected:
            image = crop_line(page.image_path, line.polygon)
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": image}, {"type": "text", "text": LINE_TRANSCRIPTION_PROMPT}],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=int(self.config.extra.get("max_new_tokens", 256)))
            generated_trimmed = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
            ]
            transcription = processor.batch_decode(generated_trimmed, skip_special_tokens=True)[0].strip()
            lines.append(
                LineOutput(
                    line_id=line.line_id,
                    polygon=line.polygon,
                    baseline=line.baseline,
                    text=transcription,
                    confidence=None,
                    reading_order=line.reading_order,
                )
            )
        output = make_output(page, self.name, self.version(), lines)
        output.metadata["budget_event"] = BudgetEvent(
            engine=self.name,
            book_id=page.book_id,
            page_id=page.page_id,
            units=len(lines),
            estimated_usd=0.0,
            description="local_qwen2vl_line_crops",
        ).to_json()
        return output

    def _run_ollama(self, page: PageRef) -> EnginePageOutput:
        model_id = self.config.model_id or self.config.model or DEFAULT_OLLAMA_MODEL
        host = str(self.config.extra.get("ollama_host", DEFAULT_OLLAMA_HOST)).rstrip("/")
        timeout_seconds = float(self.config.extra.get("timeout_seconds", 180))
        options = ollama_options(self.config.extra)
        max_image_side = int(self.config.extra.get("max_image_side", 1280))
        detected = self._line_boxes(page)
        lines: list[LineOutput] = []
        for line in detected:
            image = prepare_ollama_image(crop_line(page.image_path, line.polygon), max_side=max_image_side)
            transcription = ollama_chat(
                host=host,
                model=model_id,
                prompt=LINE_TRANSCRIPTION_PROMPT,
                image_base64=image_to_base64_jpeg(image),
                options=options,
                timeout_seconds=timeout_seconds,
            )
            lines.append(
                LineOutput(
                    line_id=line.line_id,
                    polygon=line.polygon,
                    baseline=line.baseline,
                    text=transcription,
                    confidence=None,
                    reading_order=line.reading_order,
                )
            )
        output = make_output(page, self.name, self.version(), lines)
        output.metadata["budget_event"] = BudgetEvent(
            engine=self.name,
            book_id=page.book_id,
            page_id=page.page_id,
            units=len(lines),
            estimated_usd=0.0,
            description="ollama_qwen_vl_line_crops",
        ).to_json()
        return output

    def _line_boxes(self, page: PageRef) -> list[LineOutput]:
        source = str(self.config.extra.get("line_source", "kraken")).lower()
        annotations_root_raw = self.config.extra.get("annotations_root")
        annotations_root = None if annotations_root_raw is None else Path(str(annotations_root_raw))
        if source in {"auto", "kcac_json", "json", "text_lines", "annotations", "annotations.lines", "lines", "legacy_lines"}:
            json_source = "auto" if source == "auto" else source
            lines = load_kcac_json_lines(page, annotations_root=annotations_root, source=json_source)
            if lines:
                return lines
            if source in {"kcac_json", "json", "text_lines", "annotations", "annotations.lines", "lines", "legacy_lines"}:
                raise EngineError(self.name, f"No KCAC JSON line boxes found for {page.image_path}", recoverable=True)
        if source not in {"auto", "kraken"}:
            raise EngineError(self.name, f"Unsupported qwen2vl line_source: {source}", recoverable=False)
        detector = KrakenEngine(self.config)
        return detector.detect_lines(page.image_path, page.page_id)


def ollama_options(extra: dict[str, Any]) -> dict[str, Any]:
    option_keys = {
        "num_ctx",
        "num_predict",
        "temperature",
        "top_p",
        "top_k",
        "repeat_penalty",
        "num_gpu",
    }
    return {key: extra[key] for key in option_keys if key in extra}


def prepare_ollama_image(image: Image.Image, *, max_side: int) -> Image.Image:
    prepared = image.convert("RGB")
    if max(prepared.size) > max_side:
        prepared.thumbnail((max_side, max_side))
    return prepared


def ollama_payload(model: str, prompt: str, image_base64: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64],
            }
        ],
    }
    if options:
        payload["options"] = options
    return payload


def ollama_chat(
    *,
    host: str,
    model: str,
    prompt: str,
    image_base64: str,
    options: dict[str, Any] | None,
    timeout_seconds: float,
) -> str:
    endpoint = f"{host.rstrip('/')}/api/chat"
    body = json.dumps(ollama_payload(model, prompt, image_base64, options)).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise EngineError("qwen2vl", f"Ollama request failed at {endpoint}: HTTP {exc.code}: {detail}", recoverable=True) from exc
    except urllib.error.URLError as exc:
        raise EngineError("qwen2vl", f"Ollama request failed at {endpoint}: {exc}", recoverable=True) from exc
    data = json.loads(raw)
    message = data.get("message", {})
    content = message.get("content", "") if isinstance(message, dict) else ""
    return str(content).strip()
