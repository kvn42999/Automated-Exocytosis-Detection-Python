from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile
from scipy import ndimage
from skimage import exposure, filters, measure


def iter_tiff_frames(path: Path) -> Iterator[np.ndarray]:
    """Yield grayscale pages from a multipage TIFF without loading the stack."""
    with tifffile.TiffFile(path) as tif:
        if not tif.pages:
            raise ValueError(f"TIFF contains no image pages: {path}")
        for page_number, page in enumerate(tif.pages, start=1):
            frame = page.asarray()
            if frame.ndim != 2:
                raise ValueError(
                    f"Expected grayscale TIFF pages; page {page_number} in {path} "
                    f"has shape {frame.shape}"
                )
            yield frame


def read_binary_mask(path: Path) -> np.ndarray:
    mask = np.asarray(tifffile.imread(path))
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a single 2D cell mask in {path}, got {mask.shape}")
    return mask > 0


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        raise ValueError("Cell-mask segmentation produced no foreground component")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def create_cell_mask(movie_path: Path, output_path: Path) -> np.ndarray:
    """Approximate Cell_Mask_Arch_fin using stack subtraction and Otsu segmentation."""
    frames = iter_tiff_frames(movie_path)
    try:
        first = np.asarray(next(frames), dtype=np.float64)
    except StopIteration as exc:
        raise ValueError(f"TIFF contains no image pages: {movie_path}") from exc

    total = first.copy()
    frame_count = 1
    for frame in frames:
        if frame.shape != first.shape:
            raise ValueError(f"Frame dimensions vary within {movie_path}")
        total += np.asarray(frame, dtype=np.float64)
        frame_count += 1

    difference = np.maximum(first - total / frame_count, 0)
    difference_u8 = np.rint(255 * _normalize(difference)).astype(np.uint8)
    median = ndimage.median_filter(difference_u8, size=(12, 12), mode="constant", cval=0)
    blurred = ndimage.gaussian_filter(median.astype(np.float32), sigma=5.0, mode="nearest")
    sharpened = np.clip(median + 15.0 * (median - blurred), 0, 255)
    threshold = filters.threshold_otsu(sharpened)
    mask = _largest_component(sharpened > threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, mask.astype(np.uint8), photometric="minisblack")
    return mask


def _normalize(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    low = float(np.min(image))
    high = float(np.max(image))
    if high <= low:
        return np.zeros(image.shape, dtype=np.float64)
    return (image - low) / (high - low)


def preprocess_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Subtract median outside-mask background and linearly scale to uint8."""
    if frame.shape != mask.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match frame shape {frame.shape}")
    background_pixels = np.asarray(frame)[~mask]
    if background_pixels.size == 0:
        raise ValueError("Cell mask covers the complete image; no background pixels remain")
    background = float(np.median(background_pixels))
    corrected = np.asarray(frame, dtype=np.float64) - background
    return np.rint(255 * _normalize(corrected)).astype(np.uint8)


def histogram_match_frame(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    matched = exposure.match_histograms(frame, reference, channel_axis=None)
    return np.clip(np.rint(matched), 0, 255).astype(np.uint8)


def temporal_difference(
    current: np.ndarray,
    previous_frames: list[np.ndarray],
    mask: np.ndarray,
) -> np.ndarray:
    """Subtract the pixelwise median of five preceding frames from current."""
    if len(previous_frames) != 5:
        raise ValueError(f"Exactly five preceding frames are required, got {len(previous_frames)}")
    if current.shape != mask.shape or any(frame.shape != current.shape for frame in previous_frames):
        raise ValueError("Current frame, previous frames, and mask must have identical shapes")

    blurred_current = ndimage.gaussian_filter(current.astype(np.float32), sigma=2, mode="nearest")
    background_stack = np.stack(
        [
            ndimage.median_filter(
                ndimage.gaussian_filter(frame.astype(np.float32), sigma=2, mode="nearest"),
                size=(3, 3),
                mode="constant",
                cval=0,
            )
            for frame in previous_frames
        ],
        axis=0,
    )
    background = np.median(background_stack, axis=0)
    difference = np.maximum(blurred_current - background, 0)
    return difference * mask


def multiscale_dog_response(image: np.ndarray) -> np.ndarray:
    """Median response across adjacent Gaussian scales, as in GaussFit.m."""
    smoothed = ndimage.gaussian_filter(image.astype(np.float32), sigma=6, mode="nearest")
    sigmas = 2.0 ** (np.arange(13, dtype=np.float64) / 3.0)
    blurs = [
        ndimage.gaussian_filter(smoothed, sigma=float(sigma), mode="nearest", truncate=5.0)
        for sigma in sigmas
    ]
    dogs = np.stack([blurs[index] - blurs[index + 1] for index in range(12)], axis=0)
    return np.median(dogs, axis=0)


def binary_blobs(frame: np.ndarray, minimum_area: int = 25) -> list[dict[str, float]]:
    """Measure connected foreground blobs, using 8-connected components."""
    labels = measure.label(frame > 0, connectivity=2)
    regions = measure.regionprops(labels)
    blobs: list[dict[str, float]] = []
    for region in regions:
        if region.area < minimum_area:
            continue
        row, col = region.centroid
        blobs.append(
            {
                "x": float(col),
                "y": float(row),
                "area": float(region.area),
                "major_axis": float(region.axis_major_length),
                "minor_axis": float(region.axis_minor_length),
                "perimeter": float(region.perimeter),
            }
        )
    return blobs


def iter_temporal_windows(frames: Iterator[np.ndarray]) -> Iterator[tuple[int, np.ndarray, list[np.ndarray]]]:
    """Yield MATLAB-one-based frame indices and five preceding frames, from frame 10."""
    history: deque[np.ndarray] = deque(maxlen=5)
    for matlab_index, frame in enumerate(frames, start=1):
        if matlab_index <= 5:
            history.append(frame)
            continue
        if matlab_index >= 10:
            yield matlab_index, frame, list(history)
        history.append(frame)
