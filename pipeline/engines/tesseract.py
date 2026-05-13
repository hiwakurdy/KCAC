from __future__ import annotations

from itertools import groupby
from typing import Any

from PIL import Image

from ..geometry import baseline_from_polygon
from ..models import EnginePageOutput, LineOutput, PageRef
from .base import BaseOcrEngine, line_id, make_output, polygon_from_bbox


class TesseractEngine(BaseOcrEngine):
    name = "tesseract"

    def version(self) -> str:
        pytesseract = self.require_import("pytesseract")
        return str(pytesseract.get_tesseract_version())

    def run(self, page: PageRef) -> EnginePageOutput:
        pytesseract = self.require_import("pytesseract")
        with Image.open(page.image_path) as image:
            config = ""
            if self.config.psm is not None:
                config = f"--psm {self.config.psm}"
            data = pytesseract.image_to_data(
                image,
                lang=self.config.lang or "ckb",
                config=config,
                output_type=pytesseract.Output.DICT,
            )

        rows: list[dict[str, Any]] = []
        for idx, text in enumerate(data.get("text", [])):
            value = str(text).strip()
            conf_raw = data.get("conf", [None])[idx]
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                conf = -1.0
            if not value or conf < 0:
                continue
            rows.append(
                {
                    "block": int(data["block_num"][idx]),
                    "par": int(data["par_num"][idx]),
                    "line": int(data["line_num"][idx]),
                    "text": value,
                    "left": float(data["left"][idx]),
                    "top": float(data["top"][idx]),
                    "width": float(data["width"][idx]),
                    "height": float(data["height"][idx]),
                    "confidence": conf / 100.0,
                }
            )

        lines: list[LineOutput] = []
        rows.sort(key=lambda row: (row["block"], row["par"], row["line"]))

        def key_func(row: dict[str, Any]) -> tuple[int, int, int]:
            return int(row["block"]), int(row["par"]), int(row["line"])

        for index, (_key, grouped) in enumerate(groupby(rows, key=key_func), start=1):
            words = list(grouped)
            left = min(float(word["left"]) for word in words)
            top = min(float(word["top"]) for word in words)
            right = max(float(word["left"]) + float(word["width"]) for word in words)
            bottom = max(float(word["top"]) + float(word["height"]) for word in words)
            polygon = polygon_from_bbox(left, top, right - left, bottom - top)
            text = " ".join(str(word["text"]) for word in words)
            confidence = sum(float(word["confidence"]) for word in words) / len(words)
            lines.append(
                LineOutput(
                    line_id=line_id(page.page_id, index),
                    polygon=polygon,
                    baseline=baseline_from_polygon(polygon),
                    text=text,
                    confidence=round(confidence, 4),
                    reading_order=index,
                )
            )
        return make_output(page, self.name, self.version(), lines)
