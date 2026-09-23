#!/usr/bin/env python3
"""Build a local CountixAV video/audio set from pre-existing Kinetics clips.

Place this script in the CountixAV project root:

    CountixAV/
    ├── orig_anns/         # CountixAV_train.csv, CountixAV_val.csv, CountixAV_test.csv
    ├── kinetics_clips/    # searched recursively; this is the only source directory
    ├── videos/            # created and populated as videos/
    ├── audios/            # created and populated as audios/
    ├── missing_videos.csv
    ├── audio_extraction_failures.csv
    └── get_countixav.py

Audio is extracted with FFmpeg as mono, 16 kHz, 16-bit PCM WAV. FFprobe is
used to distinguish audiovisual, video-only, audio-only, and unreadable files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import wave
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
AUDIO_REPORT_COLUMNS = (
    "split",
    "video_id",
    "kinetics_start",
    "kinetics_end",
    "source_path",
    "audio_path",
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
    audio_streams: int
    video_codecs: tuple[str, ...]
    audio_codecs: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy CountixAV clips from kinetics_clips and extract robust WAV audio."
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
        help="Only source directory, searched recursively (default: <script_dir>/kinetics_clips)",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=SCRIPT_DIR / "videos",
        help="Video output directory (default: <script_dir>/videos)",
    )
    parser.add_argument(
        "--audios-dir",
        type=Path,
        default=SCRIPT_DIR / "audios",
        help="Audio output directory (default: <script_dir>/audios)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="CSV report directory (default: <script_dir>)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Splits to process (default: train val test)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="WAV sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace valid video and audio outputs that already exist",
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
        f"CountixAV_{split}.csv",
        f"countixav_{split}.csv",
        f"countixAV_{split}.csv",
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
            f"Expected one CSV for split '{split}' in {directory}; found {len(matches)}."
        )
    return matches[0]


def read_requests(csv_path: Path, split: str) -> tuple[list[ClipRequest], int]:
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
            request = ClipRequest(split, key)
            unique.setdefault(key, request)
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
        "format=duration:stream=index,codec_type,codec_name,duration",
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
    audio_codecs = tuple(
        str(stream.get("codec_name", "unknown"))
        for stream in streams
        if stream.get("codec_type") == "audio"
    )
    duration_value = payload.get("format", {}).get("duration")
    try:
        duration = float(duration_value) if duration_value is not None else None
    except (TypeError, ValueError):
        duration = None
    return MediaProbe(
        duration=duration,
        video_streams=len(video_codecs),
        audio_streams=len(audio_codecs),
        video_codecs=video_codecs,
        audio_codecs=audio_codecs,
    )


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.copying{destination.suffix}"
    )
    if temporary.exists():
        temporary.unlink()
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def valid_wav(path: Path, sample_rate: int) -> bool:
    if not path.is_file() or path.stat().st_size <= 44:
        return False
    try:
        with wave.open(str(path), "rb") as handle:
            return (
                handle.getnchannels() == 1
                and handle.getsampwidth() == 2
                and handle.getframerate() == sample_rate
                and handle.getnframes() > 0
            )
    except (OSError, wave.Error):
        return False


def extract_wav(
    source: Path,
    destination: Path,
    ffmpeg: str,
    sample_rate: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.extracting.wav")
    if temporary.exists():
        temporary.unlink()
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
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {compact_error(result.stderr)}")
        if not valid_wav(temporary, sample_rate):
            raise RuntimeError("ffmpeg produced an invalid WAV file")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def report_base(request: ClipRequest) -> dict[str, str]:
    return {
        "split": request.split,
        "video_id": request.key.video_id,
        "kinetics_start": format_number(request.key.start),
        "kinetics_end": format_number(request.key.end),
    }


def write_report(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.sample_rate <= 0:
        raise SystemExit("--sample-rate must be greater than zero")
    if not args.annotations_dir.is_dir():
        raise SystemExit(f"Annotations directory not found: {args.annotations_dir}")
    if not args.kinetics_clips_dir.is_dir():
        raise SystemExit(
            f"Kinetics clips directory not found: {args.kinetics_clips_dir}"
        )

    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if ffprobe is None:
        raise SystemExit("ffprobe was not found in PATH")
    if ffmpeg is None:
        raise SystemExit("ffmpeg was not found in PATH")

    requests: list[ClipRequest] = []
    annotation_rows = 0
    for split in args.splits:
        csv_path = find_split_csv(args.annotations_dir, split)
        split_requests, split_rows = read_requests(csv_path, split)
        requests.extend(split_requests)
        annotation_rows += split_rows
        print(
            f"{split}: {split_rows:,} annotation rows, "
            f"{len(split_requests):,} unique clips ({csv_path.name})"
        )

    required_keys = {request_lookup_key(request) for request in requests}
    source_index, _ = index_kinetics_clips(
        args.kinetics_clips_dir,
        required_keys,
    )

    args.videos_dir.mkdir(parents=True, exist_ok=True)
    args.audios_dir.mkdir(parents=True, exist_ok=True)
    missing_video_report = args.reports_dir / "missing_videos.csv"
    audio_failure_report = args.reports_dir / "audio_extraction_failures.csv"
    video_failures: list[dict[str, str]] = []
    audio_failures: list[dict[str, str]] = []
    copied_videos = 0
    skipped_videos = 0
    extracted_audios = 0
    skipped_audios = 0

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
            probe = probe_media(source, ffprobe)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            video_failures.append(
                {
                    **report_base(request),
                    "expected_filename": f"{stem}{source.suffix.lower()}",
                    "source_path": str(source),
                    "reason": reason,
                }
            )
            audio_failures.append(
                {
                    **report_base(request),
                    "source_path": str(source),
                    "audio_path": "",
                    "reason": reason,
                }
            )
            print(f"{prefix}: unreadable source: {exc}")
            continue

        video_destination = (
            args.videos_dir / f"{stem}{source.suffix.lower()}"
        )
        if probe.video_streams == 0:
            video_failures.append(
                {
                    **report_base(request),
                    "expected_filename": video_destination.name,
                    "source_path": str(source),
                    "reason": "matching file has no video stream",
                }
            )
        else:
            try:
                if video_destination.is_file() and not args.overwrite:
                    destination_probe = probe_media(video_destination, ffprobe)
                    if destination_probe.video_streams > 0:
                        skipped_videos += 1
                    else:
                        atomic_copy(source, video_destination)
                        copied_videos += 1
                else:
                    atomic_copy(source, video_destination)
                    copied_videos += 1
            except Exception as exc:
                video_failures.append(
                    {
                        **report_base(request),
                        "expected_filename": video_destination.name,
                        "source_path": str(source),
                        "reason": f"copy failed: {type(exc).__name__}: {exc}",
                    }
                )

        audio_destination = args.audios_dir / f"{stem}.wav"
        if probe.audio_streams == 0:
            audio_failures.append(
                {
                    **report_base(request),
                    "source_path": str(source),
                    "audio_path": str(audio_destination),
                    "reason": "source has no audio stream",
                }
            )
        else:
            try:
                if valid_wav(audio_destination, args.sample_rate) and not args.overwrite:
                    skipped_audios += 1
                else:
                    extract_wav(
                        source,
                        audio_destination,
                        ffmpeg,
                        args.sample_rate,
                    )
                    extracted_audios += 1
            except Exception as exc:
                audio_failures.append(
                    {
                        **report_base(request),
                        "source_path": str(source),
                        "audio_path": str(audio_destination),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )

        status = []
        status.append("video" if probe.video_streams else "no-video")
        status.append("audio" if probe.audio_streams else "no-audio")
        print(
            f"{prefix}: {', '.join(status)} "
            f"(duration={probe.duration if probe.duration is not None else 'unknown'}s)"
        )

    write_report(missing_video_report, VIDEO_REPORT_COLUMNS, video_failures)
    write_report(audio_failure_report, AUDIO_REPORT_COLUMNS, audio_failures)

    print()
    print(f"Annotation rows: {annotation_rows:,}")
    print(f"Unique split/clip requests: {len(requests):,}")
    print(f"Videos copied now: {copied_videos:,}")
    print(f"Videos already valid: {skipped_videos:,}")
    print(f"Audio files extracted now: {extracted_audios:,}")
    print(f"Audio files already valid: {skipped_audios:,}")
    print(f"Missing or invalid videos: {len(video_failures):,} ({missing_video_report})")
    print(f"Audio extraction failures: {len(audio_failures):,} ({audio_failure_report})")
    return 0 if not video_failures and not audio_failures else 2


if __name__ == "__main__":
    sys.exit(main())