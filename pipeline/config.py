from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in incomplete environments.
    raise ModuleNotFoundError(
        "Missing dependency 'PyYAML'. Activate the project environment and run: "
        "python -m pip install -r requirements.txt"
    ) from exc


@dataclass(slots=True)
class EngineConfig:
    enabled: bool = True
    model: str | None = None
    model_id: str | None = None
    lang: str | None = None
    psm: int | None = None
    use_local_gpu: bool = False
    rate_limit_rpm: int | None = None
    monthly_budget_usd: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineConfig:
    config_path: Path
    books_jsonl: Path
    images_root: Path
    output_root: Path
    engines: dict[str, EngineConfig]
    iou_threshold: float = 0.5
    auto_accept_min_agreeing_engines: int = 3
    auto_accept_max_pairwise_edits: int = 0
    near_agreement_max_edits: int = 2
    strip_tashkeel: bool = True
    unicode_form: str = "NFC"
    test_book_ids: list[str] = field(default_factory=list)
    val_book_ids: list[str] = field(default_factory=list)
    daily_report: bool = True


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _engine_config(raw: dict[str, Any]) -> EngineConfig:
    known = {
        "enabled",
        "model",
        "model_id",
        "lang",
        "psm",
        "use_local_gpu",
        "rate_limit_rpm",
        "monthly_budget_usd",
    }
    return EngineConfig(
        enabled=bool(raw.get("enabled", True)),
        model=raw.get("model"),
        model_id=raw.get("model_id"),
        lang=raw.get("lang"),
        psm=raw.get("psm"),
        use_local_gpu=bool(raw.get("use_local_gpu", False)),
        rate_limit_rpm=raw.get("rate_limit_rpm"),
        monthly_budget_usd=raw.get("monthly_budget_usd"),
        extra={k: v for k, v in raw.items() if k not in known},
    )


def _resolve_engine_model_paths(base: Path, engines: dict[str, EngineConfig]) -> None:
    for engine_name in ("kraken", "calamari"):
        config = engines.get(engine_name)
        if config is not None and config.model:
            config.model = str(_resolve(base, config.model))


def load_config(path: Path) -> PipelineConfig:
    config_path = path.resolve()
    base = config_path.parent
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    input_cfg = raw.get("input", {})
    output_cfg = raw.get("output", {})
    consensus_cfg = raw.get("consensus", {})
    normalisation_cfg = raw.get("normalisation", {})
    split_cfg = raw.get("splits", {})
    dashboards_cfg = raw.get("dashboards", {})

    engines = {
        name: _engine_config(dict(value or {}))
        for name, value in dict(raw.get("engines", {})).items()
    }
    _resolve_engine_model_paths(base, engines)
    return PipelineConfig(
        config_path=config_path,
        books_jsonl=_resolve(base, input_cfg.get("books_jsonl", "input/books.jsonl")),
        images_root=_resolve(base, input_cfg.get("images_root", "input/images")),
        output_root=_resolve(base, output_cfg.get("root", "output")),
        engines=engines,
        iou_threshold=float(consensus_cfg.get("iou_threshold_for_line_alignment", 0.5)),
        auto_accept_min_agreeing_engines=int(consensus_cfg.get("auto_accept_min_agreeing_engines", 3)),
        auto_accept_max_pairwise_edits=int(consensus_cfg.get("auto_accept_max_pairwise_edits", 0)),
        near_agreement_max_edits=int(consensus_cfg.get("near_agreement_max_edits", 2)),
        strip_tashkeel=bool(normalisation_cfg.get("strip_tashkeel", True)),
        unicode_form=str(normalisation_cfg.get("unicode_form", "NFC")),
        test_book_ids=[str(item) for item in split_cfg.get("test_book_ids", [])],
        val_book_ids=[str(item) for item in split_cfg.get("val_book_ids", [])],
        daily_report=bool(dashboards_cfg.get("daily_report", True)),
    )
