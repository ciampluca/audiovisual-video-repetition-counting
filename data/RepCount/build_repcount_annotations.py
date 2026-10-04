#!/usr/bin/env python3
"""Process RepCount annotation CSVs, validate video metadata via ffprobe,

and generate standardized dataset annotations and error logs.
"""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Paths relative to the script directory
SCRIPT_DIR = Path(__file__).resolve().parent
ORIG_ANNS_DIR = SCRIPT_DIR / "orig_anns"
ANNOTATIONS_DIR = SCRIPT_DIR / "annotations"
VIDEOS_DIR = SCRIPT_DIR / "videos"
LOG_CSV = SCRIPT_DIR / "missing_videos.csv"

SPLITS = ("train", "val", "test")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def get_video_info(video_path: Path, ffprobe_path: str) -> tuple[float, float, int]:
    """Extract FPS, total duration (seconds), and total frame count using ffprobe."""
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,duration,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for '{video_path.name}': {result.stderr.strip()}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [{}])
    if not streams:
        raise RuntimeError(f"No video stream found in '{video_path.name}'")

    stream = streams[0]

    # Calculate exact FPS
    r_frame_rate = stream.get("r_frame_rate", "0/0")
    if "/" in r_frame_rate:
        num, den = r_frame_rate.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    else:
        fps = float(r_frame_rate)

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS ({fps}) extracted for '{video_path.name}'")

    # Determine total video duration
    duration_str = stream.get("duration") or data.get("format", {}).get("duration")
    if not duration_str:
        raise RuntimeError(f"Could not determine video duration for '{video_path.name}'")
    duration_sec = float(duration_str)

    # Determine total frame count
    nb_frames_str = stream.get("nb_frames")
    if nb_frames_str and nb_frames_str.isdigit():
        total_frames = int(nb_frames_str)
    else:
        total_frames = int(round(duration_sec * fps))

    return round(fps, 2), round(duration_sec, 2), total_frames


def extract_l_frames(row: dict) -> list[int]:
    """Extract ordered non-empty L_i frame values from an annotation row."""
    l_values = []
    i = 1
    while f"L{i}" in row:
        val = row[f"L{i}"]
        if val is not None and str(val).strip() != "":
            try:
                l_values.append(int(float(str(val).strip())))
            except ValueError:
                break
        else:
            # Stop if consecutive empty L_i is reached
            if any(row.get(f"L{j}") for j in range(i + 1, i + 5)):
                i += 1
                continue
            break
        i += 1
    return l_values


def validate_annotations(
    l_values: list[int], duration_sec: float, fps: float, total_frames: int
) -> str | None:
    """Validate L_i frame ordering and duration boundaries.

    Returns None if valid, or an error message string if invalid.
    """
    if not l_values:
        return "No L_i repetition frames found"

    # Check non-decreasing frame order
    for idx in range(len(l_values) - 1):
        if l_values[idx] > l_values[idx + 1]:
            return f"Non-ascending L_i sequence detected: L{idx+1}={l_values[idx]} > L{idx+2}={l_values[idx+1]}"

    # Check upper frame / duration bound
    last_frame = l_values[-1]
    last_frame_sec = last_frame / fps
    if last_frame_sec > (duration_sec + 1.0) and last_frame > (total_frames + int(fps)):
        return f"Last frame L{len(l_values)} ({last_frame} = {last_frame_sec:.2f}s) exceeds video duration ({duration_sec:.2f}s)"

    return None


def process_split(
    split: str, ffprobe_path: str, log_writer: csv.writer
) -> tuple[int, int]:
    """Process a single split (train/val/test) CSV file."""
    input_csv = ORIG_ANNS_DIR / f"repcount_{split}.csv"
    output_csv = ANNOTATIONS_DIR / f"repcount_{split}.csv"

    if not input_csv.is_file():
        print(f"[{split.upper()}] Input file missing: {input_csv.name}")
        return 0, 0

    processed_rows = []
    max_rep_pairs = 0
    valid_count = 0
    skipped_count = 0

    with input_csv.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        for row_idx, row in enumerate(reader, start=2):
            video_name = row.get("name", "").strip()
            cls_type = row.get("type", "").strip()
            count_val = row.get("count", "").strip()

            if not video_name:
                log_writer.writerow([split, f"Row_{row_idx}", "Missing video name in CSV"])
                skipped_count += 1
                continue

            video_path = VIDEOS_DIR / video_name
            if not video_path.is_file():
                log_writer.writerow([split, video_name, "Video file does not exist in 'videos/' folder"])
                skipped_count += 1
                continue

            # Extract video properties via ffprobe
            try:
                fps, video_end_sec, total_frames = get_video_info(video_path, ffprobe_path)
            except Exception as err:
                log_writer.writerow([split, video_name, f"ffprobe error: {err}"])
                skipped_count += 1
                continue

            # Parse and validate L_i frame annotations
            l_values = extract_l_frames(row)
            validation_error = validate_annotations(l_values, video_end_sec, fps, total_frames)
            if validation_error:
                log_writer.writerow([split, video_name, f"Annotation check failed: {validation_error}"])
                skipped_count += 1
                continue

            # Calculate segment start and end metadata
            rep_start_frame = l_values[0]
            rep_end_frame = l_values[-1]
            rep_start_sec = round(rep_start_frame / fps, 2)
            rep_end_sec = round(rep_end_frame / fps, 2)

            # Build repetition pairs (L1-L2 -> rep_1, L3-L4 -> rep_2, ...)
            rep_pairs = {}
            pair_idx = 1
            for k in range(0, len(l_values) - 1, 2):
                rep_pairs[f"rep_{pair_idx}_start_frame"] = l_values[k]
                rep_pairs[f"rep_{pair_idx}_end_frame"] = l_values[k + 1]
                pair_idx += 1

            num_pairs = len(rep_pairs) // 2
            max_rep_pairs = max(max_rep_pairs, num_pairs)

            processed_rows.append(
                {
                    "video_name": video_name,
                    "count": count_val,
                    "class": cls_type,
                    "fps": fps,
                    "video_start_sec": 0,
                    "video_end_sec": video_end_sec,
                    "repetition_segment_start_sec": rep_start_sec,
                    "repetition_segment_end_sec": rep_end_sec,
                    "repetition_segment_start_frame": rep_start_frame,
                    "repetition_segment_end_frame": rep_end_frame,
                    **rep_pairs,
                }
            )
            valid_count += 1

    # Define base and dynamic repetition fieldnames
    fieldnames = [
        "video_name",
        "count",
        "class",
        "fps",
        "video_start_sec",
        "video_end_sec",
        "repetition_segment_start_sec",
        "repetition_segment_end_sec",
        "repetition_segment_start_frame",
        "repetition_segment_end_frame",
    ]
    for i in range(1, max_rep_pairs + 1):
        fieldnames.extend([f"rep_{i}_start_frame", f"rep_{i}_end_frame"])

    # Write output CSV
    with output_csv.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for p_row in processed_rows:
            writer.writerow(p_row)

    print(f"[{split.upper()}] Processed successfully -> Saved {valid_count} rows to '{output_csv.name}' (Skipped/Logged: {skipped_count})")
    return valid_count, skipped_count


def main() -> int:
    if not ORIG_ANNS_DIR.is_dir():
        print(f"Error: Directory 'orig_anns' does not exist at path:\n{ORIG_ANNS_DIR}", file=sys.stderr)
        return 1

    if not VIDEOS_DIR.is_dir():
        print(f"Error: Directory 'videos' does not exist at path:\n{VIDEOS_DIR}", file=sys.stderr)
        return 1

    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        print("Error: 'ffprobe' executable not found in system PATH. Please install ffmpeg.", file=sys.stderr)
        return 1

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize log file for missing videos and invalid rows
    with LOG_CSV.open("w", encoding="utf-8", newline="") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["split", "video_name", "reason"])

        total_valid = 0
        total_skipped = 0

        for split in SPLITS:
            valid, skipped = process_split(split, ffprobe_path, log_writer)
            total_valid += valid
            total_skipped += skipped

    print(f"\nExecution finished! Total valid rows written: {total_valid}. Total logged issues: {total_skipped}.")
    print(f"Log saved to: '{LOG_CSV.name}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())