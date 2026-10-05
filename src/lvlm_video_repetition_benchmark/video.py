from __future__ import annotations

import base64
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoSample:
    video_data_uri: str
    source_fps: float
    frames_indices: list[int]
    total_num_frames: int
    duration: float
    sampled_frame_count: int
    candidate_frame_count: int | None = None
    sampling_mode: str = "fps"

    def media_io_kwargs(self) -> dict[str, object]:
        return {
            "fps": self.source_fps,
            "num_frames": -1,
            "frames_indices": self.frames_indices,
            "total_num_frames": self.total_num_frames,
            "duration": self.duration,
            "do_sample_frames": False,
        }


def split_jpeg_frames(payload: bytes) -> list[bytes]:
    frames: list[bytes] = []
    cursor = 0
    while True:
        start = payload.find(b"\xff\xd8", cursor)
        if start < 0:
            break
        end = payload.find(b"\xff\xd9", start + 2)
        if end < 0:
            raise ValueError("FFmpeg returned a truncated JPEG frame")
        frames.append(payload[start : end + 2])
        cursor = end + 2
    return frames


def _rate(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        divisor = float(denominator)
        return float(numerator) / divisor if divisor else 0.0
    return float(value)


def _uniform_sample_indices(candidate_count: int, sample_count: int) -> list[int]:
    if sample_count == 1:
        return [(candidate_count - 1) // 2]
    return [
        index * (candidate_count - 1) // (sample_count - 1)
        for index in range(sample_count)
    ]


def _frame_timestamps(stderr: bytes) -> list[float]:
    return [
        float(value)
        for value in re.findall(rb"pts_time:([0-9]+(?:\.[0-9]+)?)", stderr)
    ]


def sample_video(video_path: Path, sampling_fps: float, max_frames: int) -> VideoSample:
    if sampling_fps <= 0:
        raise ValueError("sampling_fps must be greater than zero")
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,nb_frames,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode:
        raise RuntimeError(f"ffprobe failed for {video_path}: {probe.stderr.strip()}")

    metadata = json.loads(probe.stdout)
    streams = metadata.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    stream = streams[0]
    source_fps = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate"))
    duration_value = stream.get("duration") or metadata.get("format", {}).get("duration")
    if source_fps <= 0 or not duration_value:
        raise RuntimeError(f"Could not determine source FPS and duration for {video_path}")
    duration = float(duration_value)
    total_num_frames = int(stream.get("nb_frames") or round(source_fps * duration))

    timestamp_pass = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "info",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-vf",
            f"fps={sampling_fps:g},showinfo",
            "-vsync",
            "0",
            "-f", "null", "-",
        ],
        capture_output=True,
        check=False,
    )
    if timestamp_pass.returncode:
        error_text = timestamp_pass.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg timestamp scan failed for {video_path}: {error_text}")

    candidate_timestamps = _frame_timestamps(timestamp_pass.stderr)
    candidate_frame_count = len(candidate_timestamps)
    if candidate_frame_count == 0:
        raise RuntimeError(f"No frames sampled from {video_path} at {sampling_fps:g} FPS")

    sampling_mode = "fps"
    selected_timestamps = candidate_timestamps
    video_filter = f"fps={sampling_fps:g},showinfo"
    if candidate_frame_count > max_frames:
        selected_indices = _uniform_sample_indices(candidate_frame_count, max_frames)
        selected_timestamps = [candidate_timestamps[index] for index in selected_indices]
        select_expression = "+".join(f"eq(n\\,{index})" for index in selected_indices)
        video_filter = (
            f"fps={sampling_fps:g},select='{select_expression}',showinfo"
        )
        sampling_mode = "uniform_fallback"
        logger.warning(
            "Video %s produced %d frames at %.3g FPS, exceeding the limit of %d; "
            "uniformly sampling across the full video.",
            video_path,
            candidate_frame_count,
            sampling_fps,
            max_frames,
        )

    sampled = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "info",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-vf",
            video_filter,
            "-frames:v",
            str(len(selected_timestamps)),
            "-vsync",
            "0",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if sampled.returncode:
        error_text = sampled.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg frame extraction failed for {video_path}: {error_text}")

    frames = split_jpeg_frames(sampled.stdout)
    if len(frames) != len(selected_timestamps):
        raise RuntimeError(
            f"FFmpeg selected {len(frames)} frames for {video_path}; "
            f"expected {len(selected_timestamps)}"
        )

    timestamps = _frame_timestamps(sampled.stderr)
    if len(timestamps) != len(frames):
        timestamps = selected_timestamps
    frames_indices = [
        min(max(round(timestamp * source_fps), 0), max(total_num_frames - 1, 0))
        for timestamp in timestamps
    ]

    encoded_frames = ",".join(base64.b64encode(frame).decode("ascii") for frame in frames)
    return VideoSample(
        video_data_uri=f"data:video/jpeg;base64,{encoded_frames}",
        source_fps=source_fps,
        frames_indices=frames_indices,
        total_num_frames=total_num_frames,
        duration=duration,
        sampled_frame_count=len(frames),
        candidate_frame_count=candidate_frame_count,
        sampling_mode=sampling_mode,
    )