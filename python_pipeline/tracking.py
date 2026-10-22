from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.optimize import linear_sum_assignment

from .image_processing import binary_blobs


@dataclass
class ConstantVelocityKalman:
    state: np.ndarray
    covariance: np.ndarray
    process_noise: np.ndarray
    measurement_noise: np.ndarray

    @classmethod
    def at(cls, x: float, y: float) -> "ConstantVelocityKalman":
        return cls(
            state=np.array([x, y, 0.0, 0.0], dtype=np.float64),
            covariance=np.diag([200.0, 200.0, 50.0, 50.0]),
            process_noise=np.diag([100.0, 100.0, 25.0, 25.0]),
            measurement_noise=np.diag([100.0, 100.0]),
        )

    @staticmethod
    def transition() -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    @staticmethod
    def observation() -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    def predict(self) -> None:
        transition = self.transition()
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + self.process_noise

    def distance(self, x: float, y: float) -> float:
        observation = self.observation()
        measurement = np.array([x, y], dtype=np.float64)
        residual = measurement - observation @ self.state
        innovation_covariance = (
            observation @ self.covariance @ observation.T + self.measurement_noise
        )
        return float(residual @ np.linalg.solve(innovation_covariance, residual))

    def correct(self, x: float, y: float) -> None:
        observation = self.observation()
        measurement = np.array([x, y], dtype=np.float64)
        innovation_covariance = (
            observation @ self.covariance @ observation.T + self.measurement_noise
        )
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        residual = measurement - observation @ self.state
        self.state = self.state + gain @ residual
        identity = np.eye(4)
        self.covariance = (
            (identity - gain @ observation)
            @ self.covariance
            @ (identity - gain @ observation).T
            + gain @ self.measurement_noise @ gain.T
        )


@dataclass
class Track:
    identifier: int
    start_frame: int
    kalman: ConstantVelocityKalman
    initial_x: float
    initial_y: float
    measurements: list[dict[str, float]] = field(default_factory=list)
    last_detection_frame: int = 0

    def update(self, blob: dict[str, float], frame_number: int) -> None:
        self.kalman.correct(blob["x"], blob["y"])
        self.last_detection_frame = frame_number
        self.measurements.append(blob)

    def result_row(self) -> dict[str, float | int]:
        return {
            "id": self.identifier,
            "centroid_1": self.initial_x,
            "centroid_2": self.initial_y,
            "time": self.start_frame,
            "lifetime": self.last_detection_frame - self.start_frame + 1,
            "visible_frames": len(self.measurements),
            "area": self.measurements[0]["area"],
            "major_axis": self.measurements[0]["major_axis"],
            "minor_axis": self.measurements[0]["minor_axis"],
            "perimeter": self.measurements[0]["perimeter"],
        }


def _assign(
    tracks: list[Track], blobs: list[dict[str, float]], unmatched_cost: float = 7.0
) -> list[tuple[int, int]]:
    """Global assignment with separate unmatched costs for tracks and detections."""
    track_count = len(tracks)
    detection_count = len(blobs)
    if track_count == 0 or detection_count == 0:
        return []

    cost = np.empty((track_count, detection_count), dtype=np.float64)
    for track_index, track in enumerate(tracks):
        for detection_index, blob in enumerate(blobs):
            cost[track_index, detection_index] = track.kalman.distance(blob["x"], blob["y"])

    size = track_count + detection_count
    augmented = np.full((size, size), 1e9, dtype=np.float64)
    augmented[:track_count, :detection_count] = cost
    for track_index in range(track_count):
        augmented[track_index, detection_count + track_index] = unmatched_cost
    for detection_index in range(detection_count):
        augmented[track_count + detection_index, detection_index] = unmatched_cost
    augmented[track_count:, detection_count:] = 0.0

    rows, columns = linear_sum_assignment(augmented)
    return [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if row < track_count
        and column < detection_count
        and cost[row, column] < 2 * unmatched_cost
    ]


def track_binary_tiff(
    binary_tiff: Path,
    output_csv: Path,
    minimum_blob_area: int = 25,
) -> pd.DataFrame:
    """Track connected components and write one row per completed event track."""
    active: list[Track] = []
    completed: list[Track] = []
    next_id = 1

    with tifffile.TiffFile(binary_tiff) as tif:
        for frame_number, page in enumerate(tif.pages, start=1):
            frame = page.asarray()
            if frame.ndim != 2:
                raise ValueError(f"Expected grayscale binary TIFF page in {binary_tiff}")
            blobs = binary_blobs(frame, minimum_area=minimum_blob_area)

            for track in active:
                track.kalman.predict()
            assignments = _assign(active, blobs)
            matched_tracks = {track_index for track_index, _ in assignments}
            matched_detections = {detection_index for _, detection_index in assignments}

            for track_index, detection_index in assignments:
                active[track_index].update(blobs[detection_index], frame_number)

            still_active: list[Track] = []
            for track_index, track in enumerate(active):
                if track_index in matched_tracks:
                    still_active.append(track)
                else:
                    completed.append(track)
            active = still_active

            for detection_index, blob in enumerate(blobs):
                if detection_index in matched_detections:
                    continue
                track = Track(
                    identifier=next_id,
                    start_frame=frame_number,
                    kalman=ConstantVelocityKalman.at(blob["x"], blob["y"]),
                    initial_x=blob["x"],
                    initial_y=blob["y"],
                    last_detection_frame=frame_number,
                )
                track.measurements.append(blob)
                active.append(track)
                next_id += 1

    completed.extend(active)
    columns = [
        "id",
        "centroid_1",
        "centroid_2",
        "time",
        "lifetime",
        "visible_frames",
        "area",
        "major_axis",
        "minor_axis",
        "perimeter",
    ]
    results = pd.DataFrame([track.result_row() for track in completed], columns=columns)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    return results
