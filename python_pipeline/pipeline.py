from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from .image_processing import (
    create_cell_mask,
    histogram_match_frame,
    iter_temporal_windows,
    iter_tiff_frames,
    multiscale_dog_response,
    preprocess_frame,
    read_binary_mask,
    temporal_difference,
)
from .tracking import track_binary_tiff


@dataclass(frozen=True)
class PipelinePaths:
    workdir: Path
    raw_dir: Path
    mask_dir: Path
    preprocessed_dir: Path
    temp_dir: Path
    processed_dir: Path
    data_dir: Path

    @classmethod
    def from_workdir(cls, workdir: Path) -> "PipelinePaths":
        workdir = workdir.expanduser().resolve()
        return cls(
            workdir=workdir,
            raw_dir=workdir / "RawData",
            mask_dir=workdir / "MaskFiles",
            preprocessed_dir=workdir / "PreProcessed",
            temp_dir=workdir / "Temp",
            processed_dir=workdir / "ProcessedFiles",
            data_dir=workdir / "DataFiles",
        )


def run_pipeline(paths: PipelinePaths, generate_masks: bool = False) -> None:
    movies = sorted(paths.raw_dir.glob("*.tif")) + sorted(paths.raw_dir.glob("*.tiff"))
    if not movies:
        raise FileNotFoundError(f"No TIFF stacks found in {paths.raw_dir}")
    for directory in (
        paths.mask_dir,
        paths.preprocessed_dir,
        paths.temp_dir,
        paths.processed_dir,
        paths.data_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    for movie_index, movie in enumerate(movies, start=1):
        mask_path = paths.mask_dir / f"{movie.stem}_mask_file.tif"
        if not mask_path.is_file():
            if not generate_masks:
                raise FileNotFoundError(
                    f"Cell mask not found for {movie.name}: {mask_path}. "
                    "Run the masks command or pass --generate-masks."
                )
            create_cell_mask(movie, mask_path)

        print(f"[{movie_index}/{len(movies)}] Preprocessing {movie.name}")
        preprocessed = paths.preprocessed_dir / f"{movie.stem}_PreProcessed.tif"
        preprocess_movie(movie, mask_path, preprocessed)

        print(f"[{movie_index}/{len(movies)}] Detecting spots in {movie.name}")
        temporary = paths.temp_dir / f"{movie.stem}_Temp.tif"
        binary = paths.processed_dir / f"{movie.stem}_Processed.tif"
        detect_movie(preprocessed, mask_path, temporary, binary)

        print(f"[{movie_index}/{len(movies)}] Tracking detections in {movie.name}")
        tracking_csv = paths.data_dir / f"{movie.stem}_Processed_tracking.csv"
        track_binary_tiff(binary, tracking_csv)

        print(f"[{movie_index}/{len(movies)}] Extracting fluorescence traces")
        trace_csv = paths.data_dir / f"{movie.stem}_PreProcessed_fluorescence_traces.csv"
        extract_fluorescence_traces(preprocessed, tracking_csv, trace_csv)
        temporary.unlink(missing_ok=True)


def preprocess_movie(movie: Path, mask_path: Path, output_path: Path) -> None:
    mask = read_binary_mask(mask_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference: np.ndarray | None = None
    frame_count = 0
    with tifffile.TiffWriter(output_path, bigtiff=True) as writer:
        for frame in iter_tiff_frames(movie):
            if frame.shape != mask.shape:
                raise ValueError(f"Mask shape {mask.shape} does not match frame shape {frame.shape}")
            converted = preprocess_frame(frame, mask)
            if reference is None:
                reference = converted
            else:
                converted = histogram_match_frame(converted, reference)
            writer.write(converted, photometric="minisblack")
            frame_count += 1
    if frame_count == 0:
        raise ValueError(f"TIFF contains no image pages: {movie}")


def detect_movie(
    preprocessed_path: Path,
    mask_path: Path,
    temporary_path: Path,
    binary_path: Path,
) -> float:
    with tifffile.TiffFile(preprocessed_path) as source:
        frame_total = len(source.pages)
    if frame_total < 10:
        raise ValueError(
            f"At least 10 frames are required for detection; got {frame_total}"
        )

    mask = read_binary_mask(mask_path)
    maxima: list[float] = []
    with tifffile.TiffWriter(temporary_path, bigtiff=True) as writer:
        windows = iter_temporal_windows(iter_tiff_frames(preprocessed_path))
        for _, current, previous in windows:
            difference = temporal_difference(current, previous, mask)
            response = multiscale_dog_response(difference)
            maxima.append(float(np.max(response)))
            writer.write(response.astype(np.float32), photometric="minisblack")
    threshold = float(np.median([0.0] * 9 + maxima))
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffFile(temporary_path) as source, tifffile.TiffWriter(
        binary_path, bigtiff=True
    ) as output:
        for page in source.pages:
            response = page.asarray()
            output.write((response > threshold).astype(np.uint8), photometric="minisblack")
    return threshold


def extract_fluorescence_traces(
    preprocessed_path: Path,
    tracking_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Sample 21x21 ROIs for each track across the 51-frame event window."""
    tracks = pd.read_csv(tracking_csv)
    trace_count = 51
    traces = np.zeros((len(tracks), trace_count), dtype=np.float64)
    requests: dict[int, list[tuple[int, int, int, int]]] = {}

    for track_index, track in tracks.iterrows():
        original_event_frame = int(track["time"]) + 9
        center_col = int(np.floor(float(track["centroid_1"]) + 0.5))
        center_row = int(np.floor(float(track["centroid_2"]) + 0.5))
        for sample_index, offset in enumerate(range(-20, 31)):
            frame_number = original_event_frame + offset
            frame_index = frame_number - 1
            if frame_index >= 0:
                requests.setdefault(frame_index, []).append(
                    (track_index, sample_index, center_row, center_col)
                )

    for frame_index, frame in enumerate(iter_tiff_frames(preprocessed_path)):
        frame_requests = requests.get(frame_index, ())
        if not frame_requests:
            continue
        padded = np.pad(frame.astype(np.float64), 10, mode="constant")
        for track_index, sample_index, row, col in frame_requests:
            patch = padded[row : row + 21, col : col + 21]
            if patch.shape != (21, 21):
                raise ValueError("Internal ROI padding error while extracting fluorescence traces")
            traces[track_index, sample_index] = float(np.mean(patch))

    result = pd.DataFrame(
        {
            "index": tracks.get("id", pd.Series(dtype=np.int64)),
            "x_pos": tracks.get("centroid_1", pd.Series(dtype=np.float64)),
            "y_pos": tracks.get("centroid_2", pd.Series(dtype=np.float64)),
            "frame_num": tracks.get("time", pd.Series(dtype=np.int64)) + 9,
        }
    )
    for sample_index in range(trace_count):
        result[f"TimePoint_{sample_index + 1}"] = traces[:, sample_index]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result
