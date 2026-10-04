#!/usr/bin/env python3
"""Build final OVR-Ego4D annotations and collect the matching videos.

Default layout, relative to this script (not to the current working directory):

    ./intermediate_anns/ovr_ego4d_{train,val,test}.csv
    ./intermediate_ego4d_videos/
    ./videos/
    ./annotations/ovr_ego4d_{train,val,test}.csv
    ./missing_ovr_ego4d_videos.csv
    ./removed_duplicate_ovr_ego4d_rows.csv

The value in ``video_id`` must be exactly equal to the source filename,
including its extension.  No ID normalization and no filename parsing are
performed.  Exactly one match is required.  The matched file is copied to
``./videos`` while preserving that exact filename.

FPS and clip frame boundaries are read only from the intermediate CSV.
``kinetics_start_sec`` and ``kinetics_end_sec`` are calculated from those
frames and the CSV FPS.  The repetition segment spans from the first frame of
the first annotated repetition to the last frame of the last annotated
repetition.  The video is not opened to obtain metadata.  Existing repetition
boundaries are only renamed; they are never redistributed or made uniform.
Final rows that are identical across every output column are deduplicated
globally in train, validation, test order.  The first copy is retained and
every removed copy is recorded in a separate CSV report.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterator


SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
MAX_REPETITIONS = 71
MEDIA_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}

REQUIRED_BASE_COLUMNS = (
    "video_id",
    "count",
    "class",
    "description",
    "fps",
    "segment_start_frame",
    "segment_end_frame",
)
REPORT_FIELDS = (
    "split",
    "csv_file",
    "csv_row",
    "video_id",
    "class",
    "matches_found",
    "candidates",
    "reason",
    "details",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy matched OVR-Ego4D videos and create the three final "
            "annotation CSV files. Default paths are relative to the script."
        )
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=SCRIPT_DIR / "intermediate_anns",
        help="input CSV directory (default: ./intermediate_anns)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=SCRIPT_DIR / "intermediate_ego4d_videos",
        help="source video directory (default: ./intermediate_ego4d_videos)",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=SCRIPT_DIR / "videos",
        help="copied-video directory (default: ./videos)",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=SCRIPT_DIR / "annotations",
        help="final annotation directory (default: ./annotations)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=SCRIPT_DIR / "missing_ovr_ego4d_videos.csv",
        help=(
            "missing/ambiguous/error report "
            "(default: ./missing_ovr_ego4d_videos.csv)"
        ),
    )
    parser.add_argument(
        "--duplicates-report",
        type=Path,
        default=SCRIPT_DIR / "removed_duplicate_ovr_ego4d_rows.csv",
        help=(
            "removed exact-duplicate report "
            "(default: ./removed_duplicate_ovr_ego4d_rows.csv)"
        ),
    )
    parser.add_argument(
        "--recursive-source",
        action="store_true",
        help="scan the source directory recursively instead of only its top level",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite video files already present in ./videos",
    )
    return parser.parse_args()


def resolve_from_script(path: Path) -> Path:
    """Resolve user-supplied relative paths from the script directory."""
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.expanduser().resolve()


def clean_header(value: str | None) -> str:
    return (value or "").lstrip("\ufeff").strip()


def clean_row(raw_row: dict[str | None, str | None]) -> dict[str, str]:
    return {
        clean_header(key): (value or "").strip()
        for key, value in raw_row.items()
        if key is not None
    }


def input_repetition_columns() -> list[str]:
    columns: list[str] = []
    for number in range(1, MAX_REPETITIONS + 1):
        columns.extend((f"rep_start_frame_{number}", f"rep_end_frame_{number}"))
    return columns


def output_columns() -> list[str]:
    columns = [
        "video_name",
        "count",
        "class",
        "description",
        "fps",
        "kinetics_start_sec",
        "kinetics_end_sec",
        "repetition_segment_start_sec",
        "repetition_segment_end_sec",
        "repetition_segment_start_frame",
        "repetition_segment_end_frame",
    ]
    for number in range(1, MAX_REPETITIONS + 1):
        columns.extend((f"rep_{number}_start_frame", f"rep_{number}_end_frame"))
    return columns


def validate_input_files(intermediate_dir: Path) -> tuple[list[Path], int, set[str]]:
    csv_paths = [
        intermediate_dir / f"ovr_ego4d_{split}.csv" for split in SPLITS
    ]
    for path in csv_paths:
        if not path.is_file():
            raise FileNotFoundError(f"input CSV not found: {path}")

    required = set(REQUIRED_BASE_COLUMNS) | set(input_repetition_columns())
    requested_ids: set[str] = set()
    total_rows = 0

    print("[1/3] Reading and validating intermediate CSV files...", flush=True)
    for path in csv_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"empty CSV or missing header: {path}")

            fields = [clean_header(field) for field in reader.fieldnames]
            if len(fields) != len(set(fields)):
                raise ValueError(
                    f"duplicate columns after header normalization: {path}"
                )
            missing = sorted(required - set(fields))
            if missing:
                raise ValueError(
                    f"{path.name} is missing columns: {', '.join(missing)}"
                )

            split_rows = 0
            for raw_row in reader:
                split_rows += 1
                total_rows += 1
                row = clean_row(raw_row)
                video_id = row.get("video_id", "")
                if video_id:
                    requested_ids.add(video_id)
            print(
                f"  {path.name}: rows={split_rows}, "
                f"distinct requested IDs so far={len(requested_ids)}",
                flush=True,
            )

    return csv_paths, total_rows, requested_ids


def iter_source_entries(source_dir: Path, recursive: bool) -> Iterator[Path]:
    if recursive:
        for root, _, files in os.walk(source_dir):
            root_path = Path(root)
            for filename in files:
                yield root_path / filename
        return

    with os.scandir(source_dir) as entries:
        for entry in entries:
            if entry.is_file(follow_symlinks=True):
                yield Path(entry.path)


def index_source_videos(
    source_dir: Path,
    requested_ids: set[str],
    recursive: bool,
) -> tuple[dict[str, list[Path]], int, int]:
    """Scan once and retain files whose exact basename appears in the CSVs."""
    index: dict[str, list[Path]] = defaultdict(list)
    scanned = 0
    valid_media = 0
    started = time.monotonic()

    mode = "recursively" if recursive else "top level only"
    print(
        f"[2/3] Indexing source videos ({mode}): {source_dir}",
        flush=True,
    )
    print("  Indexing started; no files are copied in this phase.", flush=True)

    for path in iter_source_entries(source_dir, recursive):
        scanned += 1
        if path.suffix.lower() in MEDIA_SUFFIXES:
            valid_media += 1
            if path.name in requested_ids:
                index[path.name].append(path)

        if scanned % 1000 == 0:
            elapsed = max(time.monotonic() - started, 0.001)
            print(
                f"\r  indexed={scanned} valid_clips={valid_media} "
                f"matched_IDs={len(index)}/{len(requested_ids)} "
                f"rate={scanned / elapsed:,.0f}/s",
                end="",
                flush=True,
            )

    if scanned >= 1000:
        print()
    print(
        f"  Index complete: files={scanned}, valid clips={valid_media}, "
        f"matched IDs={len(index)}/{len(requested_ids)}",
        flush=True,
    )

    for matches in index.values():
        matches.sort(key=lambda item: (item.name.lower(), str(item)))
    return dict(index), scanned, valid_media


def parse_positive_fps(fps_text: str) -> Decimal:
    try:
        fps = Decimal(fps_text)
    except InvalidOperation as exc:
        raise ValueError("fps is not numeric") from exc
    if not fps.is_finite() or fps <= 0:
        raise ValueError("fps must be finite and positive")
    return fps


def format_seconds(frame: int, fps: Decimal) -> str:
    value = (Decimal(frame) / fps).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )
    text = format(value, "f").rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def kinetics_values(row: dict[str, str], fps: Decimal) -> tuple[str, str]:
    start_frame = int(decimal_frame(row["segment_start_frame"], "segment_start_frame"))
    end_frame = int(decimal_frame(row["segment_end_frame"], "segment_end_frame"))

    if end_frame <= start_frame:
        raise ValueError("segment_end_frame must be greater than segment_start_frame")
    if start_frame < 0:
        raise ValueError("segment_start_frame cannot be negative")

    return format_seconds(start_frame, fps), format_seconds(end_frame, fps)


def decimal_frame(value: str, column: str) -> str:
    """Validate a frame value and preserve integer frame coordinates."""
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{column} is not numeric: {value!r}") from exc
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise ValueError(f"{column} is not an integer frame: {value!r}")
    return str(int(parsed))


def extract_repetitions(
    row: dict[str, str],
) -> tuple[dict[str, str], int, int]:
    repetitions: dict[str, str] = {}
    seen_blank_pair = False
    first_start_frame: int | None = None
    last_end_frame: int | None = None

    for number in range(1, MAX_REPETITIONS + 1):
        input_start = f"rep_start_frame_{number}"
        input_end = f"rep_end_frame_{number}"
        output_start = f"rep_{number}_start_frame"
        output_end = f"rep_{number}_end_frame"
        start = row.get(input_start, "")
        end = row.get(input_end, "")

        if bool(start) != bool(end):
            raise ValueError(f"incomplete repetition pair {number}")
        if not start:
            seen_blank_pair = True
            repetitions[output_start] = ""
            repetitions[output_end] = ""
            continue
        if seen_blank_pair:
            raise ValueError(f"repetition pair {number} appears after an empty pair")

        start_frame = decimal_frame(start, input_start)
        end_frame = decimal_frame(end, input_end)
        if int(end_frame) < int(start_frame):
            raise ValueError(f"repetition pair {number} ends before it starts")
        repetitions[output_start] = start_frame
        repetitions[output_end] = end_frame
        if first_start_frame is None:
            first_start_frame = int(start_frame)
        last_end_frame = int(end_frame)

    for number in range(MAX_REPETITIONS + 1, 101):
        if row.get(f"rep_start_frame_{number}", "") or row.get(
            f"rep_end_frame_{number}", ""
        ):
            raise ValueError(
                f"non-empty repetition {number} exceeds the final 71-pair schema"
            )

    if first_start_frame is None or last_end_frame is None:
        raise ValueError("no complete repetition pairs found")

    return repetitions, first_start_frame, last_end_frame


def transform_row(row: dict[str, str], video_path: Path) -> dict[str, str]:
    for column in REQUIRED_BASE_COLUMNS:
        if not row.get(column, ""):
            raise ValueError(f"missing required value: {column}")

    fps = parse_positive_fps(row["fps"])
    kinetics_start_sec, kinetics_end_sec = kinetics_values(row, fps)
    repetitions, repetition_start_frame, repetition_end_frame = (
        extract_repetitions(row)
    )
    output = {
        "video_name": video_path.name,
        "count": row["count"],
        "class": row["class"],
        "description": row["description"],
        "fps": row["fps"],
        "kinetics_start_sec": kinetics_start_sec,
        "kinetics_end_sec": kinetics_end_sec,
        "repetition_segment_start_sec": format_seconds(
            repetition_start_frame, fps
        ),
        "repetition_segment_end_sec": format_seconds(
            repetition_end_frame, fps
        ),
        "repetition_segment_start_frame": str(repetition_start_frame),
        "repetition_segment_end_frame": str(repetition_end_frame),
    }
    output.update(repetitions)
    return output


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.copying")
    temporary.unlink(missing_ok=True)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_video_copied(
    source: Path,
    videos_dir: Path,
    overwrite: bool,
) -> str:
    destination = videos_dir / source.name
    if destination.exists() and not overwrite:
        if not destination.is_file():
            raise IsADirectoryError(
                f"destination exists but is not a file: {destination}"
            )
        return "already_present"
    atomic_copy(source, destination)
    return "copied"


def issue_row(
    *,
    split: str,
    csv_file: str,
    csv_row: int,
    row: dict[str, str],
    matches: list[Path],
    reason: str,
    details: str = "",
) -> dict[str, str]:
    video_id = row.get("video_id", "")
    return {
        "split": split,
        "csv_file": csv_file,
        "csv_row": str(csv_row),
        "video_id": video_id,
        "class": row.get("class", ""),
        "matches_found": str(len(matches)),
        "candidates": " | ".join(str(item) for item in matches),
        "reason": reason,
        "details": details,
    }


def draw_progress(
    current: int,
    total: int,
    *,
    kept: int,
    copied: int,
    present: int,
    duplicates: int,
    skipped: int,
) -> None:
    width = 32
    ratio = current / total if total else 1.0
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r  [{bar}] {current}/{total} ({ratio * 100:6.2f}%) "
        f"kept={kept} copied={copied} present={present} "
        f"duplicates={duplicates} skipped={skipped}",
        end="",
        flush=True,
    )
    if current >= total:
        print()


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_duplicates_report(
    path: Path,
    rows: list[dict[str, str]],
    output_fields: list[str],
) -> None:
    fields = [
        "kept_split",
        "kept_csv_file",
        "kept_csv_row",
        "removed_split",
        "removed_csv_file",
        "removed_csv_row",
        *output_fields,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    intermediate_dir = resolve_from_script(args.intermediate_dir)
    source_dir = resolve_from_script(args.source_dir)
    videos_dir = resolve_from_script(args.videos_dir)
    annotations_dir = resolve_from_script(args.annotations_dir)
    report_path = resolve_from_script(args.report)
    duplicates_report_path = resolve_from_script(args.duplicates_report)

    if not intermediate_dir.is_dir():
        print(
            f"Error: intermediate annotation directory not found: {intermediate_dir}",
            file=sys.stderr,
        )
        return 1
    if not source_dir.is_dir():
        print(
            f"Error: source video directory not found: {source_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        csv_paths, total_rows, requested_ids = validate_input_files(
            intermediate_dir
        )
        source_index, scanned_files, valid_media = index_source_videos(
            source_dir,
            requested_ids,
            args.recursive_source,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    videos_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    print("[3/3] Copying videos and writing final annotation CSV files...", flush=True)
    issues: list[dict[str, str]] = []
    removed_duplicates: list[dict[str, str]] = []
    output_fields = output_columns()
    seen_rows: dict[tuple[str, ...], tuple[str, str, int]] = {}
    processed = 0
    kept_total = 0
    copied_total = 0
    present_total = 0
    duplicate_total = 0
    split_counts: dict[str, tuple[int, int, int]] = {}
    draw_progress(
        0,
        total_rows,
        kept=0,
        copied=0,
        present=0,
        duplicates=0,
        skipped=0,
    )

    for split, input_path in zip(SPLITS, csv_paths):
        output_path = annotations_dir / f"ovr_ego4d_{split}.csv"
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        split_kept = 0
        split_duplicates = 0
        split_skipped = 0

        with input_path.open("r", encoding="utf-8-sig", newline="") as infile, \
             temporary.open("w", encoding="utf-8", newline="") as outfile:
            reader = csv.DictReader(infile)
            writer = csv.DictWriter(
                outfile,
                fieldnames=output_fields,
                extrasaction="ignore",
            )
            writer.writeheader()

            for row_number, raw_row in enumerate(reader, start=2):
                row = clean_row(raw_row)
                video_id = row.get("video_id", "")
                matches = source_index.get(video_id, [])
                final_row: dict[str, str] | None = None
                reason = ""
                details = ""

                if not video_id:
                    reason = "missing_video_id"
                elif not matches:
                    reason = "video_id_not_found"
                elif len(matches) > 1:
                    reason = "multiple_videos_for_video_id"
                else:
                    try:
                        final_row = transform_row(row, matches[0])
                    except ValueError as exc:
                        reason = "invalid_annotation"
                        details = str(exc)

                if reason:
                    issues.append(
                        issue_row(
                            split=split,
                            csv_file=input_path.name,
                            csv_row=row_number,
                            row=row,
                            matches=matches,
                            reason=reason,
                            details=details,
                        )
                    )
                    split_skipped += 1
                else:
                    assert final_row is not None
                    row_key = tuple(final_row[field] for field in output_fields)
                    kept_location = seen_rows.get(row_key)

                    if kept_location is not None:
                        kept_split, kept_csv_file, kept_csv_row = kept_location
                        removed_duplicates.append(
                            {
                                "kept_split": kept_split,
                                "kept_csv_file": kept_csv_file,
                                "kept_csv_row": str(kept_csv_row),
                                "removed_split": split,
                                "removed_csv_file": input_path.name,
                                "removed_csv_row": str(row_number),
                                **final_row,
                            }
                        )
                        split_duplicates += 1
                        duplicate_total += 1
                    else:
                        try:
                            copy_status = ensure_video_copied(
                                matches[0],
                                videos_dir,
                                args.overwrite,
                            )
                        except OSError as exc:
                            reason = "copy_error"
                            details = f"{type(exc).__name__}: {exc}"
                            issues.append(
                                issue_row(
                                    split=split,
                                    csv_file=input_path.name,
                                    csv_row=row_number,
                                    row=row,
                                    matches=matches,
                                    reason=reason,
                                    details=details,
                                )
                            )
                            split_skipped += 1
                        else:
                            writer.writerow(final_row)
                            seen_rows[row_key] = (
                                split,
                                input_path.name,
                                row_number,
                            )
                            split_kept += 1
                            kept_total += 1
                            if copy_status == "copied":
                                copied_total += 1
                            else:
                                present_total += 1

                processed += 1
                draw_progress(
                    processed,
                    total_rows,
                    kept=kept_total,
                    copied=copied_total,
                    present=present_total,
                    duplicates=duplicate_total,
                    skipped=len(issues),
                )

        temporary.replace(output_path)
        split_counts[split] = (
            split_kept,
            split_duplicates,
            split_skipped,
        )

    write_report(report_path, issues)
    write_duplicates_report(
        duplicates_report_path,
        removed_duplicates,
        output_fields,
    )

    print("\n--- Summary ---")
    print(f"Source files scanned: {scanned_files}")
    print(f"Media files found: {valid_media}")
    print(f"Intermediate annotation rows: {total_rows}")
    for split in SPLITS:
        kept, duplicates, skipped = split_counts[split]
        print(
            f"{split}: kept={kept}, duplicates_removed={duplicates}, "
            f"skipped={skipped}"
        )
    print(f"Videos copied: {copied_total}")
    print(f"Videos already present: {present_total}")
    print(f"Final annotations: {annotations_dir}")
    print(f"Missing/ambiguous/error report: {report_path}")
    print(f"Removed exact-duplicate report: {duplicates_report_path}")

    return 0 if not issues else 2


if __name__ == "__main__":
    sys.exit(main())
