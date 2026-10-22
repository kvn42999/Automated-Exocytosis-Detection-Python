# Python exocytosis-detection pipeline

This folder ports the MATLAB workflow in `AutoDetectionExocytosis_JCB_current`.
It reads grayscale multipage TIFFs, creates or consumes one binary cell mask per
movie, detects transient spots, tracks connected components, and extracts fixed
centroid-centered fluorescence traces.

## Install

Use Python 3.10 or newer:

```powershell
python -m pip install -r python_pipeline\requirements.txt
```

## Run

Place input TIFF stacks in `RawData` and run from the repository root:

```powershell
python -m python_pipeline run --workdir .
```

The runner expects `MaskFiles\<movie-name>_mask_file.tif` for every movie. To
generate missing masks from the first frame and stack average:

```powershell
python -m python_pipeline run --workdir . --generate-masks
```

To create masks only:

```powershell
python -m python_pipeline masks --workdir .
```

For a guided, single-stack workflow in VS Code, open
`python_pipeline/analyze_one_tiff.ipynb`, select a Python kernel with the
requirements installed, and edit `TIFF_PATH` in the first code cell. You can
optionally set `MASK_PATH` to an existing mask; otherwise the notebook generates
one and writes the analysis outputs into a TIFF-specific folder beside the
input stack.

Outputs are written to `PreProcessed`, `Temp`, `ProcessedFiles`, and
`DataFiles` under the work directory. Intermediate TIFFs use the MATLAB names;
the CSVs contain named columns for track IDs, centroids, frame numbers, and
measurements. Frame numbers in CSVs are one-based to match MATLAB. Trace frames
are centered on the track's first detection and contain 20 preceding frames,
the event frame, and 30 following frames.

The port retains the original processing scales and thresholds. It uses a
constant-velocity Kalman filter and global distance assignment for tracking.
Tracks are finalized once they miss a frame and tracks still active at the end
of a movie are also written. Regions touching the image boundary are
zero-padded during trace extraction so every trace has the same 51 samples.

`visualization.py` provides an optional overlay helper; visualization is not
part of the batch pipeline.
