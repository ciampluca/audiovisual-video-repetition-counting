#!/usr/bin/env python3
"""Copy only the video stream from trimmed Countix MP4s."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent


def is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def remux_video_only(source: Path, destination: Path, ffmpeg: str) -> None:
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.tmp{destination.suffix}"
    )
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "copy",
        str(temporary),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not is_nonempty_file(temporary):
            raise RuntimeError(
                f"FFmpeg failed for {source}: {(result.stderr or '').strip()}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy the video stream from videos_with_audio/ into videos/."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=SCRIPT_DIR,
        help="Countix dataset directory (default: script directory)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("videos_with_audio"),
        help="Input directory, relative to --root unless absolute",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("videos"),
        help="Output directory, relative to --root unless absolute",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate outputs even when a non-empty video already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned remuxes without creating files or directories",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    input_dir = args.input_dir if args.input_dir.is_absolute() else root / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        return 1

    sources = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    jobs = [
        (source, output_dir / source.name)
        for source in sources
        if args.overwrite or not is_nonempty_file(output_dir / source.name)
    ]
    skipped = len(sources) - len(jobs)
    print(
        f"Found {len(sources)} MP4 files; "
        f"{len(jobs)} to remux, {skipped} existing outputs to skip."
    )
    if args.dry_run:
        print("Dry run only; no output files were created.")
        return 0

    if not jobs:
        print(f"Video-only directory: {output_dir}")
        return 0

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("Error: ffmpeg was not found in PATH.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    try:
        with tqdm(
            total=len(sources),
            initial=skipped,
            desc="Countix video-only",
            unit="video",
        ) as progress:
            for source, destination in jobs:
                remux_video_only(source, destination, ffmpeg)
                completed += 1
                progress.update(1)
    except (OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Complete: created {completed} video-only files; "
        f"skipped {skipped} existing files."
    )
    print(f"Video-only directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())