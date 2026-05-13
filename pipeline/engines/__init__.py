from __future__ import annotations

from typing import Protocol

from ..config import EngineConfig
from .base import BaseOcrEngine, EngineError
from .calamari import CalamariEngine
from .claude import ClaudeEngine
from .kraken import KrakenEngine
from .qwen2vl import Qwen2VlEngine
from .tesseract import TesseractEngine


class OcrEngineFactory(Protocol):
    def __call__(self, config: EngineConfig) -> BaseOcrEngine: ...


ENGINE_CLASSES: dict[str, OcrEngineFactory] = {
    "tesseract": TesseractEngine,
    "kraken": KrakenEngine,
    "calamari": CalamariEngine,
    "qwen2vl": Qwen2VlEngine,
    "claude": ClaudeEngine,
}

__all__ = [
    "ENGINE_CLASSES",
    "BaseOcrEngine",
    "CalamariEngine",
    "ClaudeEngine",
    "EngineError",
    "KrakenEngine",
    "OcrEngineFactory",
    "Qwen2VlEngine",
    "TesseractEngine",
]
