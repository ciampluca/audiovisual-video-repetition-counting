from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from tqdm import tqdm


DATASET_DIR = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
REP_FRAME_COLUMN = re.compile(r"^rep_(\d+)_(start|end)_frame$")
SEGMENT_COLUMNS = (
    "repetition_segment_start_sec",
    "repetition_segment_end_sec",
    "repetition_segment_start_frame",
    "repetition_segment_end_frame",
)


@dataclass(frozen=True)
class Crop:
    source_path: Path
    output_path: Path
    start_sec: Decimal
    end_sec: Decimal


@dataclass(frozen=True)
class SplitPlan:
    split: str
    source_csv: Path
    output_csv: Path
    source_fieldnames: list[str]
    output_fieldnames: list[str]
    repetition_pairs: dict[int, tuple[str, str]]
    row_count: int


def parse_decimal(value: str, column: str, row_number: int) -> Decimal:
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(
            f"Invalid {column} at CSV row {row_number}: {value!r}"
        ) from exc
    if not number.is_finite():
        raise ValueError(f"Non-finite {column} at CSV row {row_number}: {value!r}")
    return number


def format_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def build_repetition_pairs(fieldnames: list[str]) -> dict[int, tuple[str, str]]:
    columns: dict[int, dict[str, str]] = {}
    for column in fieldnames:
        match = REP_FRAME_COLUMN.fullmatch(column)
        if match:
            repetition_number = int(match.group(1))
            columns.setdefault(repetition_number, {})[match.group(2)] = column

    pairs: dict[int, tuple[str, str]] = {}
    for repetition_number, pair_columns in columns.items():
        if set(pair_columns) != {"start", "end"}:
            raise ValueError(
                f"Incomplete frame columns for repetition {repetition_number}"
            )
        pairs[repetition_number] = (
            pair_columns["start"],
            pair_columns["end"],
        )
    return pairs


def transform_annotation_row(
    row: dict[str, str],
    fieldnames: list[str],
    repetition_pairs: dict[int, tuple[str, str]],
    row_number: int,
) -> dict[str, str]:
    missing = set(("video_name", "fps", *SEGMENT_COLUMNS)).difference(fieldnames)
    if missing:
        raise ValueError(
            "Missing required annotation columns: " + ", ".join(sorted(missing))
        )

    source_name = (row.get("video_name") or "").strip()
    if not source_name or Path(source_name).name != source_name:
        raise ValueError(f"Invalid video_name at CSV row {row_number}: {source_name!r}")

    fps = parse_decimal(row["fps"], "fps", row_number)
    start_sec = parse_decimal(
        row["repetition_segment_start_sec"], SEGMENT_COLUMNS[0], row_number
    )
    end_sec = parse_decimal(
        row["repetition_segment_end_sec"], SEGMENT_COLUMNS[1], row_number
    )
    start_frame = parse_decimal(
        row["repetition_segment_start_frame"], SEGMENT_COLUMNS[2], row_number
    )
    end_frame = parse_decimal(
        row["repetition_segment_end_frame"], SEGMENT_COLUMNS[3], row_number
    )
    if fps <= 0:
        raise ValueError(f"fps must be positive at CSV row {row_number}")
    if start_sec < 0 or end_sec <= start_sec:
        raise ValueError(f"Invalid repetition segment seconds at CSV row {row_number}")
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError(f"Invalid repetition segment frames at CSV row {row_number}")

    transformed = dict(row)
    source_stem = Path(source_name).stem
    transformed["video_name"] = (
        f"{source_stem}_{format_decimal(start_sec)}_"
        f"{format_decimal(end_sec)}.mp4"
    )
    transformed["repetition_segment_start_sec"] = "0.0"
    transformed["repetition_segment_end_sec"] = format_decimal(end_sec - start_sec)
    transformed["repetition_segment_start_frame"] = "0"
    transformed["repetition_segment_end_frame"] = format_decimal(
        end_frame - start_frame
    )

    for repetition_number, (start_column, end_column) in repetition_pairs.items():
        start_value = (row.get(start_column) or "").strip()
        end_value = (row.get(end_column) or "").strip()
        if bool(start_value) != bool(end_value):
            raise ValueError(
                f"Incomplete frame interval for repetition {repetition_number} "
                f"at CSV row {row_number}"
            )
        if not start_value:
            continue

        repetition_start = parse_decimal(start_value, start_column, row_number)
        repetition_end = parse_decimal(end_value, end_column, row_number)
        if (
            repetition_start < start_frame
            or repetition_end > end_frame
            or repetition_end < repetition_start
        ):
            raise ValueError(
                f"Repetition {repetition_number} frames fall outside the segment "
                f"at CSV row {row_number}"
            )
        transformed[start_column] = format_decimal(repetition_start - start_frame)
        transformed[end_column] = format_decimal(repetition_end - start_frame)

    transformed.pop("video_start_sec", None)
    transformed.pop("video_end_sec", None)
    return transformed


def inspect_split(
    split: str,
    source_annotations_dir: Path,
    source_videos_dir: Path,
    output_annotations_dir: Path,
    output_videos_dir: Path,
    crops: dict[Path, Crop],
) -> SplitPlan:
    source_csv = source_annotations_dir / f"Countix_{split}.csv"
    output_csv = output_annotations_dir / source_csv.name
    row_count = 0

    with source_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV has no header: {source_csv}")
        repetition_pairs = build_repetition_pairs(fieldnames)
        for row_number, row in enumerate(reader, start=2):
            transformed = transform_annotation_row(
                row, fieldnames, repetition_pairs, row_number
            )
            source_path = source_videos_dir / row["video_name"].strip()
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Source video not found at CSV row {row_number}: {source_path}"
                )

            start_sec = parse_decimal(
                row["repetition_segment_start_sec"], SEGMENT_COLUMNS[0], row_number
            )
            end_sec = parse_decimal(
                row["repetition_segment_end_sec"], SEGMENT_COLUMNS[1], row_number
            )
            output_path = output_videos_dir / transformed["video_name"]
            crop = Crop(source_path, output_path, start_sec, end_sec)
            previous_crop = crops.get(output_path)
            if previous_crop is not None and previous_crop != crop:
                raise ValueError(
                    f"Output name maps to different source crops: {output_path}"
                )
            crops[output_path] = crop
            row_count += 1

    output_fieldnames = [
        column
        for column in fieldnames
        if column not in {"video_start_sec", "video_end_sec"}
    ]
    return SplitPlan(
        split,
        source_csv,
        output_csv,
        fieldnames,
        output_fieldnames,
        repetition_pairs,
        row_count,
    )


def create_crop(crop: Crop) -> None:
    temporary_path = crop.output_path.with_name(
        f".{crop.output_path.stem}.{os.getpid()}.tmp.mp4"
    )
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        format_decimal(crop.start_sec),
        "-i",
        str(crop.source_path),
        "-t",
        format_decimal(crop.end_sec - crop.start_sec),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-threads:v",
        "1",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not temporary_path.is_file():
            raise RuntimeError(
                f"ffmpeg failed for {crop.source_path} "
                f"[{crop.start_sec}, {crop.end_sec}]: {result.stderr.strip()}"
            )
        if temporary_path.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced an empty crop: {crop.output_path}")
        temporary_path.replace(crop.output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def crop_is_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def process_split(plan: SplitPlan) -> None:
    temporary_csv = plan.output_csv.with_name(
        f".{plan.output_csv.name}.{os.getpid()}.tmp"
    )
    try:
        with (
            plan.source_csv.open("r", encoding="utf-8-sig", newline="") as source_file,
            temporary_csv.open("w", encoding="utf-8", newline="") as output_file,
        ):
            reader = csv.DictReader(source_file)
            writer = csv.DictWriter(output_file, fieldnames=plan.output_fieldnames)
            writer.writeheader()
            for row_number, row in enumerate(reader, start=2):
                transformed = transform_annotation_row(
                    row,
                    plan.source_fieldnames,
                    plan.repetition_pairs,
                    row_number,
                )
                writer.writerow(transformed)
        temporary_csv.replace(plan.output_csv)
    finally:
        temporary_csv.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create Countix videos and annotations trimmed to repetition segments."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Countix directory containing full_clip_annotations and full_clip_videos",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all inputs and report outputs without creating files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate cropped videos even when a non-empty output already exists",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of videos to crop concurrently (default: up to 4)",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    dataset_dir = args.dataset_dir.resolve()
    source_annotations_dir = dataset_dir / "full_clip_annotations"
    source_videos_dir = dataset_dir / "full_clip_videos"
    output_annotations_dir = dataset_dir / "annotations"
    output_videos_dir = dataset_dir / "videos_with_audio"
    crops: dict[Path, Crop] = {}
    plans = [
        inspect_split(
            split,
            source_annotations_dir,
            source_videos_dir,
            output_annotations_dir,
            output_videos_dir,
            crops,
        )
        for split in SPLITS
    ]

    total_rows = sum(plan.row_count for plan in plans)
    print(
        f"Preflight passed: {total_rows} annotation rows, "
        f"{len(crops)} unique crops across {len(plans)} splits."
    )
    if args.dry_run:
        print("Dry run only; no output files were created.")
        return 0

    crop_jobs = [
        crop
        for crop in crops.values()
        if args.overwrite or not crop_is_complete(crop.output_path)
    ]
    reused_crops = len(crops) - len(crop_jobs)
    output_annotations_dir.mkdir(parents=True, exist_ok=True)
    output_videos_dir.mkdir(parents=True, exist_ok=True)
    with tqdm(
        total=len(crops),
        initial=reused_crops,
        desc="Countix crops",
        unit="crop",
    ) as progress:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(create_crop, crop) for crop in crop_jobs]
            for future in as_completed(futures):
                future.result()
                progress.update(1)
    for plan in plans:
        process_split(plan)
    print(
        f"Complete: wrote {total_rows} annotation rows; "
        f"created {len(crop_jobs)} crops and reused {reused_crops} existing crops."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc