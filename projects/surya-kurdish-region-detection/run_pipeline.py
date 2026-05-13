"""Command-line pipeline for Kurdish Sorani text-region auto-annotation with Surya."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback before requirements install.
    def tqdm(iterable: Any, **_: Any) -> Any:
        """Return the iterable unchanged when tqdm is not installed."""
        return iterable

try:
    from . import config
    from .utils.evaluate import evaluate_batch, print_report, save_report
    from .utils.export import build_coco, export_per_image_json, save_coco
    from .utils.preprocess import batch_preprocess, load_image
    from .utils.visualize import save_panel_viz
except ImportError:  # pragma: no cover - used when run as python run_pipeline.py.
    import config  # type: ignore
    from utils.evaluate import evaluate_batch, print_report, save_report  # type: ignore
    from utils.export import build_coco, export_per_image_json, save_coco  # type: ignore
    from utils.preprocess import batch_preprocess, load_image  # type: ignore
    from utils.visualize import save_panel_viz  # type: ignore

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the Surya annotation pipeline."""
    parser = argparse.ArgumentParser(description="Kurdish Sorani text-region detection with Surya")
    parser.add_argument("--image-dir", type=Path, default=config.IMAGE_DIR, help="Directory containing input images")
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR, help="Directory for results")
    parser.add_argument("--sample", type=int, default=config.SAMPLE_SIZE, help="Process only the first N images")
    parser.add_argument("--conf", type=float, default=config.CONF_THRESH, help="Detection confidence threshold")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Outer image batch size")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip denoise, deskew, and upscaling")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization JPGs")
    parser.add_argument("--no-coco", action="store_true", help="Skip COCO JSON export")
    parser.add_argument("--per-image-json", action="store_true", help="Export one compact JSON file per image")
    return parser.parse_args()


def setup_logging(output_dir: Path) -> Path:
    """Configure stream and file logging and return the pipeline log path."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    return log_path


def collect_image_paths(image_dir: Path, sample: int | None = None) -> list[Path]:
    """Collect sorted image paths with supported extensions and exit if none are found."""
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in config.IMG_EXTS)
    if sample is not None:
        paths = paths[: max(0, sample)]
    if not paths:
        logger.error("No images found in %s", image_dir)
        raise SystemExit(1)
    logger.info("Found %d images in %s", len(paths), image_dir)
    return paths


def _log_gpu() -> None:
    """Log CUDA device name and total VRAM when PyTorch can see a GPU."""
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch is not installed yet; install requirements.txt before running Surya")
        return

    logger.info("Configured device: %s", config.DEVICE)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_vram_gb = props.total_memory / (1024**3)
        logger.info("GPU: %s | VRAM: %.1f GB", torch.cuda.get_device_name(0), total_vram_gb)
    else:
        logger.info("CUDA is not available; running on CPU will be much slower")


def _validate_surya_dependency_versions() -> None:
    """Fail early on known Surya dependency combinations that break layout loading."""
    try:
        transformers_version = metadata.version("transformers")
    except metadata.PackageNotFoundError:
        return

    major_text = transformers_version.split(".", maxsplit=1)[0]
    try:
        major_version = int(major_text)
    except ValueError:
        return

    if major_version >= 5:
        raise RuntimeError(
            "Surya layout loading is currently incompatible with transformers>=5 in this pipeline. "
            f"Installed transformers={transformers_version}. Run: "
            "python -m pip install \"transformers>=4.56.1,<5\" --force-reinstall"
        )


def load_surya_models() -> tuple[Any, Any, Any, Any]:
    """Load Surya detection and layout models, supporting both legacy and current APIs."""
    os.environ.setdefault("TORCH_DEVICE", config.DEVICE)
    _log_gpu()
    _validate_surya_dependency_versions()

    try:
        from surya.model.detection.model import load_model as load_det_model
        from surya.model.detection.model import load_processor as load_det_proc
        from surya.model.layout.model import load_model as load_lay_model
        from surya.model.layout.processor import load_processor as load_lay_proc
    except (ImportError, ModuleNotFoundError):
        logger.info("Legacy Surya model API not found; using current predictor API")
        try:
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.layout import LayoutPredictor
            from surya.settings import settings
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("Surya is not installed. Run: pip install -r requirements.txt") from exc

        start = time.time()
        logger.info("Loading Surya detection predictor")
        det_model = DetectionPredictor()
        det_proc = None
        logger.info("Detection predictor ready in %.1fs", time.time() - start)

        start = time.time()
        logger.info("Loading Surya layout predictor")
        foundation_predictor = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        lay_model = LayoutPredictor(foundation_predictor)
        lay_proc = None
        logger.info("Layout predictor ready in %.1fs", time.time() - start)
        return det_model, det_proc, lay_model, lay_proc

    start = time.time()
    logger.info("Loading Surya detection model")
    det_model = load_det_model()
    det_proc = load_det_proc()
    logger.info("Detection model ready in %.1fs", time.time() - start)

    start = time.time()
    logger.info("Loading Surya layout model")
    lay_model = load_lay_model()
    lay_proc = load_lay_proc()
    logger.info("Layout model ready in %.1fs", time.time() - start)
    return det_model, det_proc, lay_model, lay_proc


def _run_current_surya_batch(batch: list[Any], det_model: Any, lay_model: Any) -> tuple[list[Any], list[Any]]:
    """Run a batch using Surya's current callable predictor objects."""
    det_out = det_model(batch)
    lay_out = lay_model(batch)
    return list(det_out), list(lay_out)


def _run_legacy_surya_batch(
    batch: list[Any],
    det_model: Any,
    det_proc: Any,
    lay_model: Any,
    lay_proc: Any,
) -> tuple[list[Any], list[Any]]:
    """Run a batch using Surya's legacy batch_text_detection and batch_layout_detection functions."""
    from surya.detection import batch_text_detection
    from surya.layout import batch_layout_detection

    det_out = batch_text_detection(batch, det_model, det_proc)
    lay_out = batch_layout_detection(batch, lay_model, lay_proc)
    return list(det_out), list(lay_out)


def run_surya(
    images: list[Any],
    det_model: Any,
    det_proc: Any,
    lay_model: Any,
    lay_proc: Any,
    batch_size: int,
) -> tuple[list[Any], list[Any]]:
    """Run Surya text detection and layout detection in batches with timing logs."""
    all_det: list[Any] = []
    all_lay: list[Any] = []
    total = len(images)
    num_batches = (total + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Running Surya", unit="batch"):
        start_idx = batch_idx * batch_size
        batch = images[start_idx : start_idx + batch_size]
        start = time.time()
        if det_proc is None and lay_proc is None:
            det_out, lay_out = _run_current_surya_batch(batch, det_model, lay_model)
        else:
            det_out, lay_out = _run_legacy_surya_batch(batch, det_model, det_proc, lay_model, lay_proc)
        elapsed = time.time() - start
        all_det.extend(det_out)
        all_lay.extend(lay_out)
        logger.info(
            "Batch %d/%d | %d images | %.2fs | %.3fs/img",
            batch_idx + 1,
            num_batches,
            len(batch),
            elapsed,
            elapsed / max(len(batch), 1),
        )

    return all_det, all_lay


def _load_raw_images(paths: list[Path]) -> tuple[list[Any], list[dict[str, Any]]]:
    """Load raw RGB images without preprocessing and return minimal metadata."""
    images = [load_image(path) for path in paths]
    metas = [
        {
            "path": str(path),
            "ops": ["raw_load"],
            "original_size": image.size,
            "final_size": image.size,
            "skew_angle": 0.0,
        }
        for path, image in zip(paths, images)
    ]
    return images, metas


def _image_data(paths: list[Path], sizes: list[tuple[int, int]]) -> list[dict[str, Any]]:
    """Build image metadata records for exporters."""
    return [
        {
            "name": path.name,
            "path": str(path),
            "width": int(width),
            "height": int(height),
        }
        for path, (width, height) in zip(paths, sizes)
    ]


def _print_final_instructions(summary: dict[str, Any], output_dir: Path) -> None:
    """Print next-step instructions based on the decision gate."""
    ann_dir = Path(output_dir) / "annotations"
    viz_dir = Path(output_dir) / "visualizations"
    print("\nFINAL DECISION")
    print("=" * 72)
    print(f"Decision: {summary['decision']}")
    print(f"GOOD rate: {summary['good_rate'] * 100:.1f}%")
    if summary["decision"] == "PROCEED":
        print(f"Next: import the COCO JSON from {ann_dir} into Label Studio or CVAT.")
        print("Suggested command: label-studio start")
    elif summary["decision"] == "PREPROCESS":
        print(f"Next: inspect JPG panels in {viz_dir}, then retry with --conf 0.30 or --no-preprocess.")
    else:
        print("Next: run Plan B, for example: python plan_b.py --engine paddle --sample 20")
    print("=" * 72)


def main() -> None:
    """Run the complete local Surya annotation pipeline."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    viz_dir = output_dir / "visualizations"
    ann_dir = output_dir / "annotations"
    log_dir = output_dir / "logs"
    for directory in (output_dir, viz_dir, ann_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(output_dir)
    logger.info("Logging to %s", log_path)
    logger.info("Starting Kurdish Sorani Surya pipeline")

    paths = collect_image_paths(Path(args.image_dir), args.sample)
    names = [path.name for path in paths]

    if args.no_preprocess:
        logger.info("Loading raw images without preprocessing")
        images, metas = _load_raw_images(paths)
    else:
        logger.info("Preprocessing images")
        images, metas = batch_preprocess(paths)
    logger.info("Prepared %d images; first metadata item: %s", len(images), metas[0] if metas else {})

    sizes = [(int(image.size[0]), int(image.size[1])) for image in images]
    det_model, det_proc, lay_model, lay_proc = load_surya_models()

    start = time.time()
    det_preds, lay_preds = run_surya(images, det_model, det_proc, lay_model, lay_proc, max(1, args.batch_size))
    elapsed = time.time() - start
    logger.info("Surya complete in %.2fs (%.3fs/img)", elapsed, elapsed / max(len(images), 1))

    results = evaluate_batch(names, sizes, det_preds, lay_preds, args.conf)
    summary = print_report(results)
    report_path = save_report(results, summary, log_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    logger.info("Evaluation report: %s", report_path)

    if not args.no_viz:
        logger.info("Saving visualizations to %s", viz_dir)
        for image, name, det_pred, lay_pred, result in tqdm(
            zip(images, names, det_preds, lay_preds, results),
            total=len(images),
            desc="Saving visualizations",
            unit="image",
        ):
            save_panel_viz(image, det_pred, lay_pred, viz_dir / f"{Path(name).stem}_viz.jpg", name, result["quality_flag"], args.conf)

    exporter_image_data = _image_data(paths, sizes)
    if not args.no_coco:
        coco = build_coco(exporter_image_data, det_preds, lay_preds, args.conf)
        coco_path = save_coco(coco, ann_dir / f"coco_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        logger.info("COCO file: %s", coco_path)

    if args.per_image_json:
        per_image_dir = ann_dir / "per_image"
        export_per_image_json(exporter_image_data, det_preds, lay_preds, per_image_dir, args.conf)

    _print_final_instructions(summary, output_dir)


if __name__ == "__main__":
    main()
