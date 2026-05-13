from __future__ import annotations

from pathlib import Path

from pipeline import doctor as doctor_module
from pipeline.config import EngineConfig, PipelineConfig, load_config


def test_load_config_resolves_kraken_and_calamari_model_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  images_root: images
output:
  root: output
engines:
  kraken:
    enabled: true
    model: models/kraken/model.mlmodel
  calamari:
    enabled: true
    model: models/calamari/model.ckpt
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.engines["kraken"].model == str(tmp_path / "models" / "kraken" / "model.mlmodel")
    assert config.engines["calamari"].model == str(tmp_path / "models" / "calamari" / "model.ckpt")


def test_doctor_accepts_kraken_line_detection_without_model(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        config_path=tmp_path / "config.yaml",
        books_jsonl=tmp_path / "books.jsonl",
        images_root=tmp_path,
        output_root=tmp_path / "output",
        engines={"kraken": EngineConfig(enabled=True, model=None)},
    )
    monkeypatch.setattr(doctor_module, "_module_available", lambda _module: True)

    results = doctor_module.run_doctor(config)

    kraken_model = next(result for result in results if result.name == "kraken_model")
    assert kraken_model.ok is True
    assert "line detection only" in kraken_model.message
