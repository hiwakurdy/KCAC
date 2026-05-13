from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from ..models import EnginePageOutput, LineOutput, PageRef
from .base import BaseOcrEngine, EngineError, crop_line, make_output
from .kraken import KrakenEngine


class CalamariEngine(BaseOcrEngine):
    name = "calamari"

    @property
    def executable(self) -> str:
        return str(self.config.extra.get("executable", "calamari-predict"))

    def version(self) -> str:
        completed = subprocess.run([self.executable, "--version"], check=False, capture_output=True, text=True)
        text = (completed.stdout or completed.stderr).strip()
        return text or self.executable

    def run(self, page: PageRef) -> EnginePageOutput:
        if not self.config.model:
            raise EngineError(self.name, "Calamari requires engines.calamari.model checkpoint path", recoverable=False)
        model_path = Path(self.config.model)
        if not model_path.exists():
            raise EngineError(
                self.name,
                f"Calamari model checkpoint not found: {model_path}",
                recoverable=False,
            )

        detector = KrakenEngine(self.config)
        detected = detector.detect_lines(page.image_path, page.page_id)
        if not detected:
            return make_output(page, self.name, self.version(), [])

        with tempfile.TemporaryDirectory(prefix="kcac_calamari_") as tmp:
            tmpdir = Path(tmp)
            crop_paths: list[Path] = []
            for index, line in enumerate(detected, start=1):
                crop_path = tmpdir / f"{index:04d}.png"
                crop_line(page.image_path, line.polygon).save(crop_path)
                crop_paths.append(crop_path)

            command = [
                self.executable,
                "--checkpoint",
                str(model_path),
                "--files",
                *[str(path) for path in crop_paths],
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                raise EngineError(self.name, completed.stderr.strip() or "calamari-predict failed")

            lines: list[LineOutput] = []
            for crop_path, line in zip(crop_paths, detected, strict=True):
                text_path = crop_path.with_suffix(crop_path.suffix + ".pred.txt")
                text = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else ""
                lines.append(
                    LineOutput(
                        line_id=line.line_id,
                        polygon=line.polygon,
                        baseline=line.baseline,
                        text=text,
                        confidence=None,
                        reading_order=line.reading_order,
                    )
                )
        return make_output(page, self.name, self.version(), lines)
