"""Image preprocessing helpers for document-style Kurdish Sorani images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(path: str | Path) -> Image.Image:
    """Load an image from disk as a RGB PIL image."""
    image_path = Path(path)
    with Image.open(image_path) as img:
        return img.convert("RGB")


def denoise(img_np: np.ndarray) -> np.ndarray:
    """Denoise a grayscale or RGB image using OpenCV fastNlMeansDenoising."""
    if img_np.ndim == 2:
        return cv2.fastNlMeansDenoising(img_np, None, h=10, templateWindowSize=7, searchWindowSize=21)

    ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    y_channel = cv2.fastNlMeansDenoising(
        y_channel,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    merged = cv2.merge((y_channel, cr_channel, cb_channel))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Binarize a grayscale image using Otsu and adaptive thresholding, then keep the cleaner mask."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    min_dim = int(min(gray.shape[:2]))
    if min_dim < 3:
        return otsu
    block_size = min(35, max(3, min_dim if min_dim % 2 == 1 else min_dim - 1))
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=11,
    )

    def mask_score(mask: np.ndarray) -> float:
        """Score a binary mask by foreground density and connected-component sanity."""
        foreground_pct = float(np.mean(mask < 128))
        if foreground_pct <= 0.0:
            return -1.0
        density_score = 1.0 - min(abs(foreground_pct - 0.12) / 0.12, 1.0)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats((mask < 128).astype(np.uint8), 8)
        if num_labels <= 1:
            return density_score - 1.0
        areas = stats[1:, cv2.CC_STAT_AREA]
        small_noise_pct = float(np.mean(areas <= 3)) if len(areas) else 1.0
        return density_score - (0.35 * small_noise_pct)

    return otsu if mask_score(otsu) >= mask_score(adaptive) else adaptive


def estimate_skew_angle(binary: np.ndarray) -> float:
    """Estimate document skew in degrees using Hough line segments."""
    if binary.ndim == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    h, w = binary.shape[:2]
    min_line_length = max(30, int(w * 0.10))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(60, int(w * 0.06)),
        minLineLength=min_line_length,
        maxLineGap=max(8, int(w * 0.01)),
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = [float(v) for v in line]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1.0:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if -15.0 <= angle <= 15.0:
            angles.append(angle)

    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew(img_np: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by the estimated skew angle while preserving canvas size."""
    if abs(angle) < 0.5:
        return img_np

    h, w = img_np.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    border_value: int | tuple[int, int, int] = 255 if img_np.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(
        img_np,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def ensure_min_resolution(img: Image.Image, min_dpi_width: int = 1200) -> Image.Image:
    """Upscale an image if its width is below the minimum useful detector width."""
    width, height = img.size
    if width >= min_dpi_width:
        return img

    scale = min_dpi_width / max(width, 1)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    logger.debug("Upscaling %s from %sx%s to %sx%s", getattr(img, "filename", "image"), width, height, *new_size)
    return img.resize(new_size, Image.Resampling.LANCZOS)


def preprocess_for_surya(
    path: str | Path,
    denoise: bool = True,
    fix_skew: bool = True,
    ensure_resolution: bool = True,
) -> tuple[Image.Image, dict[str, Any]]:
    """Load and optionally upscale, denoise, binarize for skew estimation, and deskew one image."""
    image_path = Path(path)
    meta: dict[str, Any] = {
        "path": str(image_path),
        "ops": [],
        "original_size": None,
        "final_size": None,
        "skew_angle": 0.0,
    }

    pil_img = load_image(image_path)
    meta["original_size"] = pil_img.size

    if ensure_resolution:
        before_size = pil_img.size
        pil_img = ensure_min_resolution(pil_img)
        if pil_img.size != before_size:
            meta["ops"].append("upscale_min_width_1200")

    img_np = np.array(pil_img)

    if denoise:
        img_np = globals()["denoise"](img_np)
        meta["ops"].append("denoise_fastnlmeans")

    skew_angle = 0.0
    if fix_skew:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np
        binary = binarize(gray)
        skew_angle = estimate_skew_angle(binary)
        if abs(skew_angle) >= 0.5:
            img_np = deskew(img_np, skew_angle)
            meta["ops"].append(f"deskew_{skew_angle:.2f}deg")

    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    meta["skew_angle"] = round(float(skew_angle), 4)
    meta["final_size"] = (int(img_np.shape[1]), int(img_np.shape[0]))
    return Image.fromarray(img_np.astype(np.uint8), mode="RGB"), meta


def batch_preprocess(
    paths: list[Path],
    denoise: bool = True,
    fix_skew: bool = True,
    ensure_resolution: bool = True,
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    """Preprocess many images, falling back to raw RGB loading per failed image."""
    images: list[Image.Image] = []
    metas: list[dict[str, Any]] = []

    for path in paths:
        try:
            image, meta = preprocess_for_surya(
                path,
                denoise=denoise,
                fix_skew=fix_skew,
                ensure_resolution=ensure_resolution,
            )
        except Exception as exc:  # pragma: no cover - depends on corrupt input files.
            logger.exception("Preprocessing failed for %s; using raw image", path)
            image = load_image(path)
            meta = {
                "path": str(Path(path)),
                "ops": ["raw_fallback"],
                "original_size": image.size,
                "final_size": image.size,
                "skew_angle": 0.0,
                "error": str(exc),
            }
        images.append(image)
        metas.append(meta)

    return images, metas
