from __future__ import annotations

import argparse
from pathlib import Path

from .image_processing import create_cell_mask
from .pipeline import PipelinePaths, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and analyze exocytosis events.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run preprocessing, detection, tracking, and traces.")
    run_parser.add_argument("--workdir", type=Path, default=Path.cwd())
    run_parser.add_argument("--generate-masks", action="store_true")

    masks_parser = subparsers.add_parser("masks", help="Generate cell masks for RawData TIFFs.")
    masks_parser.add_argument("--workdir", type=Path, default=Path.cwd())

    args = parser.parse_args()
    paths = PipelinePaths.from_workdir(args.workdir)
    if args.command == "masks":
        paths.mask_dir.mkdir(parents=True, exist_ok=True)
        movies = sorted(paths.raw_dir.glob("*.tif")) + sorted(paths.raw_dir.glob("*.tiff"))
        if not movies:
            parser.error(f"No TIFF stacks found in {paths.raw_dir}")
        for movie in movies:
            output = paths.mask_dir / f"{movie.stem}_mask_file.tif"
            create_cell_mask(movie, output)
            print(f"Created mask: {output}")
        return

    run_pipeline(paths, generate_masks=args.generate_masks)


if __name__ == "__main__":
    main()
