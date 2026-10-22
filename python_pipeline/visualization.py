from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import draw, exposure

from .image_processing import iter_tiff_frames


def visualize_tracks(
    movie_path: Path,
    tracking_csv: Path,
    output_path: Path,
    frame_number: int = 1,
    radius: int = 10,
) -> np.ndarray:
    """Write a red-circle overlay for track centroids on one one-based TIFF frame."""
    if frame_number < 1:
        raise ValueError("frame_number must be one-based and greater than zero")
    frame = None
    for index, candidate in enumerate(iter_tiff_frames(movie_path), start=1):
        if index == frame_number:
            frame = candidate
            break
    if frame is None:
        raise IndexError(f"Frame {frame_number} does not exist in {movie_path}")

    display = exposure.rescale_intensity(frame, out_range=np.uint8).astype(np.uint8)
    rgb = np.repeat(display[:, :, None], 3, axis=2)
    tracks = pd.read_csv(tracking_csv)
    for _, track in tracks.iterrows():
        row = int(np.floor(float(track["centroid_2"]) + 0.5))
        col = int(np.floor(float(track["centroid_1"]) + 0.5))
        rr, cc = draw.circle_perimeter(row, col, radius, shape=frame.shape)
        rgb[rr, cc] = np.array([255, 0, 0], dtype=np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_path)
    return rgb
