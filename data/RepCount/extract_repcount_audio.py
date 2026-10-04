#!/usr/bin/env python3
"""Extract valid WAV audio from referenced RepCount dataset videos.

The script reads ``repcount_*.csv`` from ``annotations/`` and resolves each exact
``video_name`` in ``videos/``, validates the first audio stream by decoding it,
and converts that stream to 16-bit PCM WAV in ``audios/``. The output keeps the
same video ID and replaces the video extension with ``.wav``.

Videos that are missing, have no audio stream, contain undecodable audio, or
cannot be exported are recorded in ``repcount_audio_issues.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


VIDEO_COLUMNS = ("video_name", "video_id", "name")
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class VideoReference:
    """Store the annotation locations that reference one video."""

    video_name: str
    annotation_files: set[str] = field(default_factory=set)
    splits: set[str] = field(default_factory=set)
    row_numbers: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and extract WAV audio from videos referenced by RepCount annotations."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=SCRIPT_DIR,
        help="Dataset root directory (default: directory containing this script).",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("annotations"),
        help="Annotation directory, relative to --root unless absolute.",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("videos"),
        help="Video directory, relative to --root unless absolute.",
    )
    parser.add_argument(
        "--audios-dir",
        type=Path,
        default=Path("audios"),
        help="Output directory, relative to --root unless absolute.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("repcount_audio_issues.csv"),
        help="Issue report, relative to --root unless absolute.",
    )
    parser.add_argument(
        "--pattern",
        default="repcount_*.csv",
        help="Glob used to select annotation CSV files (default: repcount_*.csv).",
    )
    return parser.parse_args()


def resolve_from_root(root: Path, path: Path) -> Path:
    """Resolve a path against the dataset root when it is not absolute."""
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def infer_split(csv_path: Path) -> str:
    """Infer train, val, or test from an annotation filename."""
    stem = csv_path.stem.lower()
    for split in ("train", "val", "test"):
        if split in stem:
            return split
    return "unknown"


def select_video_column(fieldnames: list[str] | None) -> str | None:
    """Return the first supported video-name column present in the CSV."""
    if not fieldnames:
        return None
    for column in VIDEO_COLUMNS:
        if column in fieldnames:
            return column
    return None


def read_video_references(
    annotations_dir: Path,
    pattern: str,
) -> tuple[dict[str, VideoReference], list[dict[str, str]]]:
    """Read unique video names and collect annotation-level issues."""
    references: dict[str, VideoReference] = {}
    issues: list[dict[str, str]] = []
    csv_files = sorted(path for path in annotations_dir.glob(pattern) if path.is_file())

    if not csv_files:
        raise FileNotFoundError(
            f"No annotation CSV files matching {pattern!r} in {annotations_dir}"
        )

    print(f"Found {len(csv_files)} annotation CSV file(s).", flush=True)

    for csv_path in csv_files:
        split = infer_split(csv_path)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            video_column = select_video_column(reader.fieldnames)
            if video_column is None:
                issues.append(
                    make_issue(
                        video_name="",
                        reason="missing_video_column",
                        details=(
                            f"Expected one of {', '.join(VIDEO_COLUMNS)} in "
                            f"{csv_path.name}"
                        ),
                        annotation_files=csv_path.name,
                        splits=split,
                        row_numbers="",
                    )
                )
                continue

            for row_number, row in enumerate(reader, start=2):
                raw_name = (row.get(video_column) or "").strip()
                if not raw_name:
                    issues.append(
                        make_issue(
                            video_name="",
                            reason="empty_video_name",
                            details=f"Empty {video_column} value",
                            annotation_files=csv_path.name,
                            splits=split,
                            row_numbers=str(row_number),
                        )
                    )
                    continue

                reference = references.setdefault(raw_name, VideoReference(raw_name))
                reference.annotation_files.add(csv_path.name)
                reference.splits.add(split)
                reference.row_numbers.append(f"{csv_path.name}:{row_number}")

    return references, issues


def run_command(command: list[str]) -> tuple[bool, str]:
    """Run a command and return its success state and compact diagnostics."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, str(exc)

    diagnostics = (result.stderr or result.stdout or "").strip()
    if len(diagnostics) > 1000:
        diagnostics = f"{diagnostics[:997]}..."
    return result.returncode == 0, diagnostics


def probe_audio_stream(video_path: Path, ffprobe: str) -> tuple[bool, str, dict]:
    """Check whether the file exposes at least one audio stream."""
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index,codec_name,channels,sample_rate,duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, "probe_failed", {"details": str(exc)}

    if result.returncode != 0:
        diagnostics = (result.stderr or result.stdout or "").strip()
        return False, "probe_failed", {"details": diagnostics[:1000]}

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return False, "probe_failed", {"details": str(exc)}

    streams = payload.get("streams") or []
    if not streams:
        return False, "no_audio_stream", {"details": "No audio stream found"}

    return True, "", streams[0]


def validate_audio(video_path: Path, ffmpeg: str) -> tuple[bool, str]:
    """Decode the complete first audio stream and fail on decoding errors."""
    command = [
        ffmpeg,
        "-v",
        "error",
        "-xerror",
        "-nostdin",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-vn",
        "-f",
        "null",
        "-",
    ]
    return run_command(command)


def extract_audio_stream(
    video_path: Path,
    output_path: Path,
    ffmpeg: str,
) -> tuple[bool, str, str]:
    """Convert the first audio stream to PCM WAV and replace atomically."""
    temporary_path = output_path.with_name(
        f".{output_path.stem}.audio_tmp.wav"
    )
    if temporary_path.exists():
        temporary_path.unlink()

    command = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "pcm_s16le",
        "-map_metadata",
        "0",
        str(temporary_path),
    ]
    success, diagnostics = run_command(command)
    if not success:
        if temporary_path.exists():
            temporary_path.unlink()
        return False, "audio_export_failed", diagnostics

    valid, validation_details = validate_audio(temporary_path, ffmpeg)
    if not valid:
        temporary_path.unlink(missing_ok=True)
        return False, "audio_decode_failed", validation_details

    temporary_path.replace(output_path)
    return True, "", ""


def make_issue(
    video_name: str,
    reason: str,
    details: str,
    annotation_files: str,
    splits: str,
    row_numbers: str,
) -> dict[str, str]:
    """Build one issue-report row."""
    return {
        "video_name": video_name,
        "reason": reason,
        "details": details,
        "annotation_files": annotation_files,
        "splits": splits,
        "annotation_rows": row_numbers,
    }


def issue_from_reference(
    reference: VideoReference,
    reason: str,
    details: str,
) -> dict[str, str]:
    """Build an issue-report row from a collected video reference."""
    return make_issue(
        video_name=reference.video_name,
        reason=reason,
        details=details,
        annotation_files=";".join(sorted(reference.annotation_files)),
        splits=";".join(sorted(reference.splits)),
        row_numbers=";".join(reference.row_numbers),
    )


def print_progress(current: int, total: int, label: str, width: int = 36) -> None:
    """Print an in-place progress bar without external dependencies."""
    ratio = current / total if total else 1.0
    completed = min(width, int(round(ratio * width)))
    bar = "#" * completed + "-" * (width - completed)
    short_label = label if len(label) <= 42 else f"...{label[-39:]}"
    print(
        f"\r[{bar}] {current:>{len(str(total))}}/{total} {short_label:<42}",
        end="",
        flush=True,
    )
    if current >= total:
        print()


def write_issue_report(report_path: Path, issues: list[dict[str, str]]) -> None:
    """Write the issue report even when no issues were found."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_name",
        "reason",
        "details",
        "annotation_files",
        "splits",
        "annotation_rows",
    ]
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(issues)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    annotations_dir = resolve_from_root(root, args.annotations_dir)
    videos_dir = resolve_from_root(root, args.videos_dir)
    audios_dir = resolve_from_root(root, args.audios_dir)
    report_path = resolve_from_root(root, args.report)

    if not annotations_dir.is_dir():
        print(f"Error: annotation directory not found: {annotations_dir}", file=sys.stderr)
        return 1
    if not videos_dir.is_dir():
        print(f"Error: video directory not found: {videos_dir}", file=sys.stderr)
        return 1

    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if ffprobe is None or ffmpeg is None:
        print("Error: ffmpeg and ffprobe must be available in PATH.", file=sys.stderr)
        return 1

    try:
        references, issues = read_video_references(annotations_dir, args.pattern)
    except (OSError, csv.Error) as exc:
        print(f"Error reading annotations: {exc}", file=sys.stderr)
        return 1

    audios_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(references)} unique referenced video(s).", flush=True)

    extracted = 0
    total = len(references)
    for current, reference in enumerate(references.values(), start=1):
        print_progress(current - 1, total, reference.video_name)

        relative_name = Path(reference.video_name)
        if relative_name.is_absolute() or relative_name.name != reference.video_name:
            issues.append(
                issue_from_reference(
                    reference,
                    "invalid_video_name",
                    "The annotation value must be a plain filename without directories",
                )
            )
            print_progress(current, total, reference.video_name)
            continue

        video_path = videos_dir / reference.video_name
        if not video_path.is_file():
            issues.append(
                issue_from_reference(reference, "video_not_found", str(video_path))
            )
            print_progress(current, total, reference.video_name)
            continue

        has_audio, reason, stream_info = probe_audio_stream(video_path, ffprobe)
        if not has_audio:
            issues.append(
                issue_from_reference(
                    reference,
                    reason,
                    str(stream_info.get("details", "")),
                )
            )
            print_progress(current, total, reference.video_name)
            continue

        output_path = audios_dir / f"{Path(reference.video_name).stem}.wav"
        exported, export_reason, export_details = extract_audio_stream(
            video_path,
            output_path,
            ffmpeg,
        )
        if not exported:
            issues.append(
                issue_from_reference(
                    reference,
                    export_reason,
                    export_details,
                )
            )
            print_progress(current, total, reference.video_name)
            continue

        extracted += 1
        print_progress(current, total, reference.video_name)

    if total == 0:
        print_progress(0, 0, "No referenced videos")

    try:
        write_issue_report(report_path, issues)
    except OSError as exc:
        print(f"Error writing issue report: {exc}", file=sys.stderr)
        return 1

    print(f"Extracted audio files: {extracted}")
    print(f"Issues recorded: {len(issues)}")
    print(f"Audio directory: {audios_dir}")
    print(f"Issue report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())