from __future__ import annotations

from pipeline.engines.qwen2vl import ollama_options, ollama_payload


def test_ollama_payload_uses_chat_images_field() -> None:
    payload = ollama_payload("qwen25vl-sorani-ocr:latest", "transcribe", "abc123", {"num_ctx": 2048})
    assert payload["model"] == "qwen25vl-sorani-ocr:latest"
    assert payload["stream"] is False
    assert payload["options"] == {"num_ctx": 2048}
    assert payload["messages"][0]["content"] == "transcribe"
    assert payload["messages"][0]["images"] == ["abc123"]


def test_ollama_options_keeps_generation_keys_only() -> None:
    assert ollama_options({"num_ctx": 1024, "temperature": 0, "backend": "ollama"}) == {
        "num_ctx": 1024,
        "temperature": 0,
    }
