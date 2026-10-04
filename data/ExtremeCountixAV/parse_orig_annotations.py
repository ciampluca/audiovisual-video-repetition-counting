#!/usr/bin/env python3

"""Convert the original ExtremeCountixAV test annotations to a dense format.



The script only parses annotations: it never crops, renames, or modifies videos.

FPS values are read from the corresponding files in the local videos directory.

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "orig_anns/ExtremeCountixAV_test.csv"
DEFAULT_VIDEOS_DIR = SCRIPT_DIR / "videos"
DEFAULT_OUTPUT = SCRIPT_DIR / "annotations/ExtremeCountixAV_test.csv"
DEFAULT_FAILURES = SCRIPT_DIR / "ExtremeCountixAV_test_parse_failures.csv"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse ExtremeCountixAV_test.csv, read video FPS, and generate "
            "uniform per-repetition frame intervals."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Original annotation CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=DEFAULT_VIDEOS_DIR,
        help=f"Directory containing the videos (default: {DEFAULT_VIDEOS_DIR})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Converted CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--failures-csv",
        type=Path,
        default=DEFAULT_FAILURES,
        help=f"Rows that could not be converted (default: {DEFAULT_FAILURES})",
    )
    parser.add_argument(
        "--max-repetitions",
        type=int,
        default=None,
        help=(
            "Optional number of rep_N_start/end column pairs. By default it "
            "is inferred from the largest number_of_repetitions in the CSV."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively inside the videos directory.",
    )
    return parser.parse_args()


def resolve_input_csv(path: Path) -> Path:
    """Accept both ExtremeCountixAV_test and ExtremeCountixAV_test.csv."""
    if path.is_file():
        return path
    if path.suffix.lower() == ".csv":
        extensionless_path = path.with_suffix("")
        if extensionless_path.is_file():
            return extensionless_path
    if not path.suffix:
        csv_path = path.with_suffix(".csv")
        if csv_path.is_file():
            return csv_path
    raise FileNotFoundError(f"Input CSV not found: {path}")


def build_video_index(videos_dir: Path, recursive: bool) -> dict[str, list[Path]]:
    if not videos_dir.is_dir():
        raise NotADirectoryError(f"Videos directory not found: {videos_dir}")

    paths: Iterable[Path]
    paths = videos_dir.rglob("*") if recursive else videos_dir.glob("*")
    index: dict[str, list[Path]] = {}
    for path in paths:
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            index.setdefault(path.stem, []).append(path)
    return index


def find_video(youtube_id: str, index: dict[str, list[Path]]) -> Path:
    """Match the full youtube_id stem; do not convert it to a Kinetics name."""
    source_name = Path(youtube_id).name
    stem = (
        Path(source_name).stem
        if Path(source_name).suffix.lower() in VIDEO_EXTENSIONS
        else source_name
    )
    matches = index.get(stem, [])
    if not matches:
        raise FileNotFoundError(f"No video with stem {stem!r} was found")
    if len(matches) > 1:
        rendered = ", ".join(str(p) for p in matches)
        raise RuntimeError(f"Multiple videos match {stem!r}: {rendered}")
    return matches[0]


def read_video_info(video_path: Path) -> tuple[float, int]:
    """Return the video FPS and total number of frames."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is not available on PATH")

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or "unknown ffprobe error"
        raise RuntimeError(f"ffprobe failed: {message}")

    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError("the file has no readable video stream")

    stream = streams[0]
    fps: float | None = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        rate = stream.get(key)
        if not rate or rate == "0/0":
            continue
        numerator, denominator = rate.split("/", maxsplit=1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            continue
        candidate_fps = float(numerator) / denominator_value
        if math.isfinite(candidate_fps) and candidate_fps > 0:
            fps = candidate_fps
            break
    if fps is None:
        raise RuntimeError("ffprobe did not return a valid FPS value")

    total_frames: int | None = None
    raw_frame_count = stream.get("nb_frames")
    if raw_frame_count not in (None, "", "N/A"):
        try:
            candidate_frames = int(raw_frame_count)
            if candidate_frames > 0:
                total_frames = candidate_frames
        except (TypeError, ValueError):
            pass

    # Some containers do not store nb_frames. In that case, infer it from
    # duration and FPS rather than decoding the entire video.
    if total_frames is None:
        raw_duration = stream.get("duration") or payload.get("format", {}).get(
            "duration"
        )
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "ffprobe returned neither nb_frames nor a valid duration"
            ) from exc
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError("ffprobe returned an invalid video duration")
        total_frames = round(duration * fps)

    if total_frames <= 0:
        raise RuntimeError("the video frame count is not positive")
    return fps, total_frames


def required_int(value: str, field_name: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}: {value!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    return int(numeric)


def uniform_boundaries(start_frame: int, end_frame: int, count: int) -> list[int]:
    """Equivalent to rounding linspace(start_frame, end_frame, count + 1)."""
    if count <= 0:
        raise ValueError(f"number_of_repetitions must be positive, got {count}")
    if end_frame <= start_frame:
        raise ValueError(
            "repetition_end_frame must be greater than repetition_start_frame"
        )

    span = end_frame - start_frame
    boundaries = [round(start_frame + span * i / count) for i in range(count + 1)]
    boundaries[0] = start_frame
    boundaries[-1] = end_frame

    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError(
            "the repetition segment is too short to create non-empty integer intervals"
        )
    return boundaries


def format_number(
    value: float, decimals: int = 6, keep_one_decimal: bool = False
) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    text = text if text else "0"
    if keep_one_decimal and "." not in text:
        text += ".0"
    return text


def output_columns(max_repetitions: int) -> list[str]:
    columns = [
        "video_name",
        "count",
        "class",
        "fps",
        "repetition_segment_start_sec",
        "repetition_segment_end_sec",
        "repetition_segment_start_frame",
        "repetition_segment_end_frame",
    ]
    for rep_index in range(1, max_repetitions + 1):
        columns.extend(
            [f"rep_{rep_index}_start_frame", f"rep_{rep_index}_end_frame"]
        )
    return columns


def infer_max_repetitions(rows: list[dict[str, str]]) -> int:
    """Return the largest valid positive count found in the input CSV."""
    valid_counts: list[int] = []
    for row in rows:
        try:
            count = required_int(
                row.get("number_of_repetitions"), "number_of_repetitions"
            )
        except ValueError:
            # The row will later be written to the failures CSV with details.
            continue
        if count > 0:
            valid_counts.append(count)

    if not valid_counts:
        raise ValueError(
            "no valid positive number_of_repetitions values were found in the CSV"
        )
    return max(valid_counts)


def convert_row(
    row: dict[str, str],
    video_index: dict[str, list[Path]],
    max_repetitions: int,
) -> dict[str, object]:
    youtube_id = (row.get("youtube_id") or "").strip()
    if not youtube_id:
        raise ValueError("empty youtube_id")

    # Some original rows have no action_class value. Mark it explicitly as
    # unknown: the class name is metadata and is not used to match videos.
    action_class = (row.get("action_class") or "").strip() or "unknown"

    source_repetition_start = required_int(
        row["repetition_start_frame"], "repetition_start_frame"
    )
    source_repetition_end = required_int(
        row["repetition_end_frame"], "repetition_end_frame"
    )
    crop_start = required_int(row["start_crop_frame"], "start_crop_frame")
    crop_end = required_int(row["end_crop_frame"], "end_crop_frame")
    count = required_int(row["number_of_repetitions"], "number_of_repetitions")
    if count > max_repetitions:
        raise ValueError(
            f"count {count} exceeds the configured maximum of {max_repetitions}"
        )

    video_path = find_video(youtube_id, video_index)
    fps, total_frames = read_video_info(video_path)

    if source_repetition_start == 0 and source_repetition_end == -1:
        repetition_start = 0
        # Frame intervals use an exclusive right boundary: a video with N
        # frames is represented by the interval [0, N].
        repetition_end = total_frames
    elif source_repetition_start < 0 or source_repetition_end < 0:
        raise ValueError(
            "the full-video sentinel must be repetition_start_frame=0 and "
            "repetition_end_frame=-1"
        )
    else:
        if crop_start < 0:
            raise ValueError(
                "start_crop_frame must be non-negative for a bounded repetition segment"
            )
        if source_repetition_start < crop_start:
            raise ValueError("repetition_start_frame precedes start_crop_frame")

        # Extreme CountixAV annotations refer to the original video. Local
        # videos begin at start_crop_frame, so every output frame coordinate
        # must be expressed relative to that crop origin.
        repetition_start = source_repetition_start - crop_start
        repetition_end = source_repetition_end - crop_start

        # end_crop_frame is preserved as source metadata, but it is not used
        # as a hard validity constraint. A few released annotations have a
        # repetition end beyond that value. The original loader similarly
        # computes coordinates from start_crop_frame and relies on the actual
        # decoded video length for the right boundary.
        if crop_end >= 0 and source_repetition_end > crop_end:
            print(
                f"WARNING: {youtube_id}: repetition_end_frame "
                f"({source_repetition_end}) exceeds end_crop_frame "
                f"({crop_end}); using the local video bounds.",
                file=sys.stderr,
            )

        if repetition_start >= total_frames:
            raise ValueError(
                "the local repetition start lies outside the available video frames"
            )
        if repetition_end > total_frames:
            print(
                f"WARNING: {youtube_id}: local repetition end "
                f"({repetition_end}) exceeds the video frame count "
                f"({total_frames}); clipping it to the video end.",
                file=sys.stderr,
            )
            repetition_end = total_frames

    boundaries = uniform_boundaries(repetition_start, repetition_end, count)

    converted: dict[str, object] = {
        # Keep the original stem and only use the extension of the matching file.
        "video_name": (
            f"{youtube_id}{video_path.suffix}"
            if not Path(youtube_id).suffix.lower() in VIDEO_EXTENSIONS
            else youtube_id
        ),
        "count": format_number(float(count), 1, keep_one_decimal=True),
        "class": action_class,
        "fps": format_number(fps, 6),
        "repetition_segment_start_sec": format_number(
            repetition_start / fps, 6, keep_one_decimal=True
        ),
        "repetition_segment_end_sec": format_number(
            repetition_end / fps, 6, keep_one_decimal=True
        ),
        "repetition_segment_start_frame": repetition_start,
        "repetition_segment_end_frame": repetition_end,
    }

    for rep_index in range(1, max_repetitions + 1):
        if rep_index <= count:
            converted[f"rep_{rep_index}_start_frame"] = boundaries[rep_index - 1]
            converted[f"rep_{rep_index}_end_frame"] = boundaries[rep_index]
        else:
            converted[f"rep_{rep_index}_start_frame"] = ""
            converted[f"rep_{rep_index}_end_frame"] = ""
    return converted


def main() -> int:
    args = parse_args()
    if args.max_repetitions is not None and args.max_repetitions <= 0:
        print("ERROR: --max-repetitions must be positive", file=sys.stderr)
        return 2

    try:
        input_csv = resolve_input_csv(args.input_csv)
        video_index = build_video_index(args.videos_dir, args.recursive)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    required_columns = {
        "youtube_id",
        "repetition_start_frame",
        "repetition_end_frame",
        "start_crop_frame",
        "end_crop_frame",
        "number_of_repetitions",
        "action_class",
    }
    converted_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    with input_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            print(
                "ERROR: missing input columns: " + ", ".join(sorted(missing_columns)),
                file=sys.stderr,
            )
            return 2
        source_rows = list(reader)

    try:
        detected_max = infer_max_repetitions(source_rows)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    max_repetitions = args.max_repetitions or detected_max
    if max_repetitions < detected_max:
        print(
            "ERROR: --max-repetitions is smaller than the largest count in the "
            f"CSV ({max_repetitions} < {detected_max})",
            file=sys.stderr,
        )
        return 2

    fieldnames = output_columns(max_repetitions)

    for row_number, row in enumerate(source_rows, start=2):
        try:
            converted_rows.append(convert_row(row, video_index, max_repetitions))
        except Exception as exc:  # Continue and report every problematic annotation.
            failures.append(
                {
                    "row_number": row_number,
                    "youtube_id": row.get("youtube_id", ""),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(converted_rows)

    args.failures_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.failures_csv.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(
            destination, fieldnames=["row_number", "youtube_id", "reason"]
        )
        writer.writeheader()
        writer.writerows(failures)

    print(f"Input annotations: {len(converted_rows) + len(failures)}")
    print(f"Maximum count:     {detected_max}")
    print(f"Repetition pairs:  {max_repetitions}")
    print(f"Converted:         {len(converted_rows)}")
    print(f"Failures:          {len(failures)}")
    print(f"Output CSV:        {args.output_csv}")
    print(f"Failures CSV:      {args.failures_csv}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())