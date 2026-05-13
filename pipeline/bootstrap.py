from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .discovery import discover_pages, raw_engine_output_path
from .engines import ENGINE_CLASSES, EngineError
from .jsonio import write_json
from .models import EngineFailure, EnginePageOutput, PageRef
from .retry import ConsecutiveFailureGuard, retry_call


def enabled_engine_names(config: PipelineConfig) -> list[str]:
    return [name for name, value in config.engines.items() if value.enabled]


def run_engine(page: PageRef, engine_name: str, config: PipelineConfig) -> EnginePageOutput:
    engine_config = config.engines[engine_name]
    engine = ENGINE_CLASSES[engine_name](engine_config)
    try:
        return retry_call(lambda: engine.run(page), attempts=3)
    except EngineError as exc:
        return EnginePageOutput(
            page_id=page.page_id,
            book_id=page.book_id,
            engine=engine_name,
            engine_version="",
            image_filename=page.image_path.name,
            failures=[
                EngineFailure(
                    engine=engine_name,
                    page_id=page.page_id,
                    error_type=exc.__class__.__name__,
                    message=str(exc),
                    recoverable=exc.recoverable,
                )
            ],
            metadata={"image_path": str(page.image_path), "width": page.width, "height": page.height},
        )
    except Exception as exc:
        return EnginePageOutput(
            page_id=page.page_id,
            book_id=page.book_id,
            engine=engine_name,
            engine_version="",
            image_filename=page.image_path.name,
            failures=[
                EngineFailure(
                    engine=engine_name,
                    page_id=page.page_id,
                    error_type=exc.__class__.__name__,
                    message=str(exc),
                    recoverable=True,
                )
            ],
            metadata={"image_path": str(page.image_path), "width": page.width, "height": page.height},
        )


def _skipped_output(page: PageRef, engine_name: str, previous_failure: EngineFailure) -> EnginePageOutput:
    return EnginePageOutput(
        page_id=page.page_id,
        book_id=page.book_id,
        engine=engine_name,
        engine_version="",
        image_filename=page.image_path.name,
        failures=[
            EngineFailure(
                engine=engine_name,
                page_id=page.page_id,
                error_type="EngineSkipped",
                message=(
                    "Skipped because this engine already had a non-recoverable setup failure: "
                    f"{previous_failure.message}"
                ),
                recoverable=False,
            )
        ],
        metadata={"image_path": str(page.image_path), "width": page.width, "height": page.height},
    )


def _failure_summary(output: EnginePageOutput) -> str:
    if not output.failures:
        return ""
    failure = output.failures[0]
    return f"{failure.error_type}: {failure.message}"


def bootstrap_pages(config: PipelineConfig, *, limit: int | None = None, force: bool = False) -> list[Path]:
    pages = discover_pages(config.images_root, limit=limit)
    written: list[Path] = []
    guards = {name: ConsecutiveFailureGuard(max_failures=5) for name in enabled_engine_names(config)}
    skipped_engines: dict[str, EngineFailure] = {}
    for page in pages:
        for engine_name in enabled_engine_names(config):
            output_path = raw_engine_output_path(config.output_root, page.book_id, page.page_id, engine_name)
            if output_path.exists() and not force:
                written.append(output_path)
                continue
            output = (
                _skipped_output(page, engine_name, skipped_engines[engine_name])
                if engine_name in skipped_engines
                else run_engine(page, engine_name, config)
            )
            write_json(output_path, output.to_json())
            written.append(output_path)
            if output.failures:
                non_recoverable = next((failure for failure in output.failures if not failure.recoverable), None)
                if non_recoverable is not None:
                    skipped_engines[engine_name] = non_recoverable
                    continue
                try:
                    guards[engine_name].record_failure()
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Stopping {engine_name} after {guards[engine_name].failures} consecutive failures. "
                        f"Last page: {page.page_id}. Last error: {_failure_summary(output)}"
                    ) from exc
            else:
                guards[engine_name].record_success()
    return written
