#!/usr/bin/env python3
"""Build a local Countix video set from pre-existing Kinetics clips.

Expected project layout::

    Countix/
    ├── get_countix.py
    ├── orig_anns/
    │   ├── Countix_train.csv
    │   ├── Countix_val.csv
    │   └── Countix_test.csv
    ├── kinetics_clips/       # searched recursively
    ├── full_clip_videos/               # created and populated
    └── missing_videos.csv    # created or replaced

The annotation CSVs identify clips through ``video_id``, ``kinetics_start``
and ``kinetics_end``. For example, video_id=dyzWet-ZFx4, start=78 and end=88
matches a clip named ``dyzWet-ZFx4_000078_000088`` with any supported media
extension.

Files are copied atomically. Existing valid outputs are skipped unless
``--overwrite`` is supplied. FFprobe is used to reject unreadable files and
files without a video stream.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_VERSION = "1.0"
SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
MEDIA_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}
REQUIRED_COLUMNS = (
    "video_id",
    "kinetics_start",
    "kinetics_end",
    "repetition_start",
    "repetition_end",
    "count",
)
VIDEO_REPORT_COLUMNS = (
    "split",
    "video_id",
    "kinetics_start",
    "kinetics_end",
    "expected_filename",
    "source_path",
    "reason",
)


@dataclass(frozen=True)
class ClipKey:
    video_id: str
    start: float
    end: float


@dataclass(frozen=True)
class ClipRequest:
    split: str
    key: ClipKey


@dataclass(frozen=True)
class MediaProbe:
    duration: float | None
    video_streams: int
    video_codecs: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy Countix clips from a recursively searched kinetics_clips "
            "directory into full_clip_videos/."
        )
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=SCRIPT_DIR / "orig_anns",
        help="Annotation directory (default: <script_dir>/orig_anns)",
    )
    parser.add_argument(
        "--kinetics-clips-dir",
        type=Path,
        default=SCRIPT_DIR / "kinetics_clips",
        help=(
            "Only source directory, searched recursively "
            "(default: <script_dir>/kinetics_clips)"
        ),
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=SCRIPT_DIR / "full_clip_videos",
        help="Video output directory (default: <script_dir>/videos)",
    )
    parser.add_argument(
        "--missing-csv",
        type=Path,
        default=SCRIPT_DIR / "missing_videos.csv",
        help="Missing-video report (default: <script_dir>/missing_videos.csv)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Splits to process (default: train val test)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace valid video outputs that already exist",
    )
    return parser.parse_args()


def clean_row(row: dict[str, str | None]) -> dict[str, str]:
    return {
        key.lstrip("\ufeff").strip(): (value or "").strip()
        for key, value in row.items()
        if key is not None
    }


def find_split_csv(directory: Path, split: str) -> Path:
    preferred = (
        f"Countix_{split}.csv",
        f"countix_{split}.csv",
        f"{split}.csv",
    )
    for filename in preferred:
        path = directory / filename
        if path.is_file():
            return path

    matches = sorted(
        path
        for path in directory.glob("*.csv")
        if split in path.stem.lower() and "local" not in path.stem.lower()
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one CSV for split {split!r} in {directory}; "
            f"found {len(matches)}."
        )
    return matches[0]


def read_requests(csv_path: Path, split: str) -> tuple[list[ClipRequest], int]:
    """Read one split and remove repeated copies of the same clip in it."""
    unique: dict[ClipKey, ClipRequest] = {}
    row_count = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: empty CSV or missing header")
        columns = {column.lstrip("\ufeff").strip() for column in reader.fieldnames}
        missing = set(REQUIRED_COLUMNS) - columns
        if missing:
            raise ValueError(
                f"{csv_path}: missing columns: {', '.join(sorted(missing))}"
            )

        for line_number, raw_row in enumerate(reader, start=2):
            row = clean_row(raw_row)
            if not row.get("video_id"):
                continue
            row_count += 1
            try:
                start = float(row["kinetics_start"])
                end = float(row["kinetics_end"])
                repetition_start = float(row["repetition_start"])
                repetition_end = float(row["repetition_end"])
                count = float(row["count"])
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path}:{line_number}: invalid numeric value"
                ) from exc

            values = (start, end, repetition_start, repetition_end, count)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{csv_path}:{line_number}: non-finite numeric value")
            if end <= start:
                raise ValueError(
                    f"{csv_path}:{line_number}: kinetics_end <= kinetics_start"
                )

            key = ClipKey(row["video_id"], start, end)
            unique.setdefault(key, ClipRequest(split, key))

    return list(unique.values()), row_count


def time_token(value: float) -> str:
    if value.is_integer():
        return f"{int(value):06d}"
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.6f}".rstrip("0").rstrip(".")


def clip_stem(key: ClipKey) -> str:
    return f"{key.video_id}_{time_token(key.start)}_{time_token(key.end)}"


def lookup_key_from_filename(filename: str) -> tuple[str, float, float] | None:
    """Parse the conventional 11-character YouTube ID Kinetics filename."""
    stem = Path(filename).stem
    if len(stem) < 12 or stem[11] != "_":
        return None
    fields = stem[12:].split("_")
    if len(fields) < 2:
        return None
    try:
        start = round(float(fields[0].replace("p", ".")), 6)
        end = round(float(fields[1].replace("p", ".")), 6)
    except ValueError:
        return None
    return stem[:11], start, end


def request_lookup_key(request: ClipRequest) -> tuple[str, float, float]:
    return (
        request.key.video_id,
        round(request.key.start, 6),
        round(request.key.end, 6),
    )


def index_kinetics_clips(
    directory: Path,
    required_keys: set[tuple[str, float, float]],
) -> tuple[dict[tuple[str, float, float], list[Path]], int]:
    index: dict[tuple[str, float, float], list[Path]] = {}
    scanned_media = 0
    print(f"Scanning {directory.resolve()} ...", flush=True)

    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        scanned_media += 1
        lookup_key = lookup_key_from_filename(path.name)
        if lookup_key in required_keys:
            index.setdefault(lookup_key, []).append(path)
        if scanned_media % 5000 == 0:
            print(
                f"  scanned {scanned_media:,} media files; "
                f"matched {len(index):,}/{len(required_keys):,} requested clips",
                flush=True,
            )

    for paths in index.values():
        paths.sort()
    print(
        f"Scan complete: {scanned_media:,} media files; "
        f"matched {len(index):,}/{len(required_keys):,} requested clips"
    )
    return index, scanned_media


def choose_source(request: ClipRequest, candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    canonical = clip_stem(request.key)
    return sorted(
        candidates,
        key=lambda path: (
            path.stem != canonical,
            path.suffix.lower() != ".mp4",
            str(path),
        ),
    )[0]


def compact_error(text: str, limit: int = 1200) -> str:
    cleaned = " ".join(text.strip().split())
    return cleaned[-limit:] if cleaned else "unknown error"


def probe_media(path: Path, ffprobe: str) -> MediaProbe:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {compact_error(result.stderr)}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON") from exc

    streams = payload.get("streams", [])
    video_codecs = tuple(
        str(stream.get("codec_name", "unknown"))
        for stream in streams
        if stream.get("codec_type") == "video"
    )
    duration_value = payload.get("format", {}).get("duration")
    try:
        duration = float(duration_value) if duration_value is not None else None
    except (TypeError, ValueError):
        duration = None

    return MediaProbe(
        duration=duration,
        video_streams=len(video_codecs),
        video_codecs=video_codecs,
    )


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.copying{destination.suffix}"
    )
    temporary.unlink(missing_ok=True)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def report_base(request: ClipRequest) -> dict[str, str]:
    return {
        "split": request.split,
        "video_id": request.key.video_id,
        "kinetics_start": format_number(request.key.start),
        "kinetics_end": format_number(request.key.end),
    }


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VIDEO_REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    annotations_dir = args.annotations_dir.expanduser().resolve()
    kinetics_clips_dir = args.kinetics_clips_dir.expanduser().resolve()
    videos_dir = args.videos_dir.expanduser().resolve()
    missing_csv = args.missing_csv.expanduser().resolve()

    if not annotations_dir.is_dir():
        raise SystemExit(f"Annotations directory not found: {annotations_dir}")
    if not kinetics_clips_dir.is_dir():
        raise SystemExit(f"Kinetics clips directory not found: {kinetics_clips_dir}")

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise SystemExit("ffprobe was not found in PATH; install FFmpeg first")

    requests: list[ClipRequest] = []
    annotation_rows = 0
    try:
        for split in args.splits:
            csv_path = find_split_csv(annotations_dir, split)
            split_requests, split_rows = read_requests(csv_path, split)
            requests.extend(split_requests)
            annotation_rows += split_rows
            print(
                f"{split}: {split_rows:,} annotation rows, "
                f"{len(split_requests):,} distinct clips ({csv_path.name})"
            )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    required_keys = {request_lookup_key(request) for request in requests}
    source_index, _ = index_kinetics_clips(kinetics_clips_dir, required_keys)

    videos_dir.mkdir(parents=True, exist_ok=True)
    video_failures: list[dict[str, str]] = []
    copied_videos = 0
    skipped_videos = 0

    for position, request in enumerate(requests, start=1):
        stem = clip_stem(request.key)
        candidates = source_index.get(request_lookup_key(request), [])
        source = choose_source(request, candidates)
        prefix = f"[{position}/{len(requests)}] {request.split}/{stem}"

        if source is None:
            video_failures.append(
                {
                    **report_base(request),
                    "expected_filename": f"{stem}.mp4",
                    "source_path": "",
                    "reason": "matching file not found in kinetics_clips",
                }
            )
            print(f"{prefix}: missing source")
            continue

        if len(candidates) > 1:
            print(f"{prefix}: warning: {len(candidates)} candidates; using {source}")

        try:
            source_probe = probe_media(source, ffprobe)
            if source_probe.video_streams == 0:
                raise ValueError("matching file has no video stream")
            if source_probe.duration is None or source_probe.duration <= 0:
                raise ValueError("matching file has no valid duration")

            destination = videos_dir / f"{stem}{source.suffix.lower()}"
            if destination.is_file() and not args.overwrite:
                destination_probe = probe_media(destination, ffprobe)
                if (
                    destination_probe.video_streams > 0
                    and destination_probe.duration is not None
                    and destination_probe.duration > 0
                ):
                    skipped_videos += 1
                    action = "already valid"
                else:
                    atomic_copy(source, destination)
                    copied_videos += 1
                    action = "replaced invalid output"
            else:
                atomic_copy(source, destination)
                copied_videos += 1
                action = "copied"

            print(
                f"{prefix}: {action} "
                f"(duration={source_probe.duration:.3f}s, "
                f"codec={','.join(source_probe.video_codecs)})"
            )
        except Exception as exc:
            video_failures.append(
                {
                    **report_base(request),
                    "expected_filename": f"{stem}{source.suffix.lower()}",
                    "source_path": str(source),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"{prefix}: failed: {exc}")

    write_report(missing_csv, video_failures)

    print()
    print("--- Summary ---")
    print(f"Annotation rows: {annotation_rows:,}")
    print(f"Distinct split/clip requests: {len(requests):,}")
    print(f"Videos copied now: {copied_videos:,}")
    print(f"Videos already valid: {skipped_videos:,}")
    print(f"Missing or invalid videos: {len(video_failures):,}")
    print(f"Missing report: {missing_csv}")

    return 0 if not video_failures else 2


if __name__ == "__main__":
    sys.exit(main())
