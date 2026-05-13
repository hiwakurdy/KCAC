from __future__ import annotations

import os

from ..models import BudgetEvent, EnginePageOutput, LineOutput, PageRef
from .base import BaseOcrEngine, EngineError, crop_line, image_to_base64_jpeg, make_output
from .kraken import KrakenEngine


class ClaudeEngine(BaseOcrEngine):
    name = "claude"

    def version(self) -> str:
        anthropic = self.require_import("anthropic")
        return f"{self.config.model or 'claude-sonnet-4-7-20260101'} via anthropic {getattr(anthropic, '__version__', '')}"

    def run(self, page: PageRef) -> EnginePageOutput:
        try:
            import anthropic
        except ImportError as exc:
            raise EngineError(self.name, "anthropic SDK is required", recoverable=False) from exc
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EngineError(self.name, "ANTHROPIC_API_KEY is required for Claude OCR", recoverable=False)

        client = anthropic.Anthropic(api_key=api_key)
        detector = KrakenEngine(self.config)
        detected = detector.detect_lines(page.image_path, page.page_id)
        lines: list[LineOutput] = []
        model = self.config.model or "claude-sonnet-4-7-20260101"
        for line in detected:
            image = crop_line(page.image_path, line.polygon)
            response = client.messages.create(
                model=model,
                max_tokens=int(self.config.extra.get("max_tokens", 256)),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_to_base64_jpeg(image),
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Transcribe this single Kurdish Sorani Arabic-script print line exactly as printed. "
                                    "Return only the transcription."
                                ),
                            },
                        ],
                    }
                ],
            )
            transcription_parts = [getattr(block, "text", "") for block in response.content]
            lines.append(
                LineOutput(
                    line_id=line.line_id,
                    polygon=line.polygon,
                    baseline=line.baseline,
                    text="".join(transcription_parts).strip(),
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
            estimated_usd=float(self.config.extra.get("estimated_page_usd", 0.0)),
            description="claude_line_crop_transcription",
        ).to_json()
        return output
