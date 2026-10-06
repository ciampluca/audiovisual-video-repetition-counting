from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
import time
from typing import Any, Callable

from tqdm import tqdm

from lvlm_video_repetition_benchmark.datasets import Annotation, load_annotations
from lvlm_video_repetition_benchmark.metrics import compute_metrics
from lvlm_video_repetition_benchmark.parsing import parse_response
from lvlm_video_repetition_benchmark.prompting import PromptContextError, render_prompt
from lvlm_video_repetition_benchmark.video import VideoSample, sample_video
from lvlm_video_repetition_benchmark.vllm_client import VLLMVideoClient


logger = logging.getLogger(__name__)


class DatasetSkippedError(RuntimeError):
    pass


PREDICTION_COLUMNS = [
    "dataset",
    "annotation_split",
    "annotation_id",
    "annotation_row",
    "video_name",
    "class_name",
    "description",
    "video_path",
    "gt_count",
    "pred_count",
    "action_description",
    "reasoning",
    "signed_error",
    "abs_error",
    "relative_abs_error",
    "raw_response",
    "status",
    "error_message",
    "seed",
    "model_name",
    "model_id",
    "prompt_name",
    "prompt_id",
    "fps",
    "sampled_frame_count",
    "candidate_frame_count",
    "sampling_mode",
    "temperature",
]

METRIC_COLUMNS = [
    "dataset",
    "annotation_split",
    "model_name",
    "prompt_name",
    "fps",
    "temperature",
    "seed",
    "run_status",
    "n_total",
    "n_valid_predictions",
    "mae_n",
    "mae_percent",
    "legacy_mae_count",
    "rmse_count",
    "obz_percent",
    "obo_percent",
]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _result_record(
    annotation: Annotation,
    seed: int,
    cfg: Any,
    raw_response: str = "",
    status: str = "error",
    error_message: str = "",
    sampled_frame_count: int | None = None,
    candidate_frame_count: int | None = None,
    sampling_mode: str | None = None,
) -> dict[str, Any]:
    return {
        "dataset": cfg.dataset.name,
        "annotation_split": annotation.annotation_split,
        "annotation_id": annotation.annotation_id,
        "annotation_row": annotation.annotation_row,
        "video_name": annotation.video_name,
        "class_name": annotation.class_name,
        "description": annotation.description,
        "video_path": str(annotation.video_path),
        "gt_count": annotation.gt_count,
        "pred_count": None,
        "action_description": None,
        "reasoning": None,
        "signed_error": None,
        "abs_error": None,
        "relative_abs_error": None,
        "raw_response": raw_response,
        "status": status,
        "error_message": error_message,
        "seed": seed,
        "model_name": cfg.model.name,
        "model_id": cfg.model.hf_id,
        "prompt_name": cfg.prompt.name,
        "prompt_id": cfg.prompt.slug,
        "fps": float(cfg.sampling.fps),
        "sampled_frame_count": sampled_frame_count,
        "candidate_frame_count": candidate_frame_count,
        "sampling_mode": sampling_mode,
        "temperature": float(cfg.generation.temperature),
    }


def _is_retryable_request_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code == 429 or (isinstance(status_code, int) and 500 <= status_code < 600):
        return True
    return type(error).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "InternalServerError",
        "PoolTimeout",
        "ReadError",
        "RemoteProtocolError",
        "TimeoutException",
        "WriteError",
    }


def _generate_with_retries(
    client: Any,
    *,
    video: VideoSample,
    prompt: str,
    seed: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str:
    attempt = 0
    while True:
        try:
            return client.generate(
                video=video,
                prompt=prompt,
                seed=seed,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if attempt >= max_retries or not _is_retryable_request_error(exc):
                raise
            delay = retry_backoff_seconds * (2**attempt)
            logger.warning(
                "Transient vLLM request error for seed %d; retry %d/%d in %.1f s: %s",
                seed,
                attempt + 1,
                max_retries,
                delay,
                exc,
            )
            if delay > 0:
                time.sleep(delay)
            attempt += 1


def _write_outputs(
    output_dir: Path,
    prediction_rows: list[dict[str, Any]],
    annotations: list[Annotation],
    seeds: list[int],
    cfg: Any,
    run_status: str,
) -> None:
    _write_csv(output_dir / "predictions.csv", PREDICTION_COLUMNS, prediction_rows)
    metric_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_rows = [row for row in prediction_rows if row["seed"] == seed]
        metrics = compute_metrics(seed_rows)
        metric_rows.append(
            {
                "dataset": cfg.dataset.name,
                "annotation_split": annotations[0].annotation_split if annotations else "",
                "model_name": cfg.model.name,
                "prompt_name": cfg.prompt.name,
                "fps": float(cfg.sampling.fps),
                "temperature": float(cfg.generation.temperature),
                "seed": seed,
                "run_status": run_status,
                "n_total": len(annotations),
                "n_valid_predictions": metrics["n"],
                **metrics,
            }
        )
    _write_csv(output_dir / "metrics_by_seed.csv", METRIC_COLUMNS, metric_rows)


def run_benchmark(
    cfg: Any,
    client: Any | None = None,
    sampler: Callable[[Path, float, int], VideoSample] = sample_video,
) -> tuple[Path, str]:
    temperature = float(cfg.generation.temperature)
    fps = float(cfg.sampling.fps)
    max_frames = int(cfg.sampling.max_frames)
    seeds = [int(seed) for seed in cfg.benchmark.seeds]
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("generation.temperature must be non-negative")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("sampling.fps must be greater than zero")
    if not seeds:
        raise ValueError("benchmark.seeds must contain at least one seed")
    runtime_cfg = getattr(cfg, "runtime", None)
    max_retries = int(getattr(runtime_cfg, "max_retries", 2))
    retry_backoff_seconds = float(getattr(runtime_cfg, "retry_backoff_seconds", 1.0))
    if max_retries < 0:
        raise ValueError("runtime.max_retries must be non-negative")
    if not math.isfinite(retry_backoff_seconds) or retry_backoff_seconds < 0:
        raise ValueError("runtime.retry_backoff_seconds must be non-negative")

    workspace_root = Path(cfg.paths.workspace_root).resolve()
    dataset_dir = workspace_root / "data" / str(cfg.dataset.folder)
    annotations, annotation_csv, split = load_annotations(
        dataset_dir,
        str(cfg.dataset.slug),
        str(cfg.dataset.annotation_prefix),
    )
    configured_context_fields = getattr(cfg.prompt, "context_fields", ()) or ()
    if isinstance(configured_context_fields, str):
        configured_context_fields = (configured_context_fields,)
    context_fields = tuple(str(field) for field in configured_context_fields)
    if context_fields and not annotations:
        raise DatasetSkippedError(
            f"Dataset {cfg.dataset.name!r} skipped for prompt {cfg.prompt.slug!r}: "
            f"selected annotation file {annotation_csv} contains no rows for the "
            f"required prompt context fields {', '.join(context_fields)}."
        )
    try:
        prompts = {
            annotation.annotation_id: render_prompt(
                str(cfg.prompt.text), annotation, context_fields
            )
            for annotation in annotations
        }
    except PromptContextError as exc:
        raise DatasetSkippedError(
            f"Dataset {cfg.dataset.name!r} skipped for prompt {cfg.prompt.slug!r}: "
            f"selected annotation file {annotation_csv} has missing or invalid "
            f"required prompt context ({exc})."
        ) from exc

    output_dir = (
        Path(cfg.paths.output_root).resolve()
        / str(cfg.dataset.slug)
        / str(cfg.model.slug)
        / "video-only"
        / str(cfg.prompt.slug)
        / f"fps-{fps:g}"
        / f"temperature-{temperature:g}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if client is None:
        client = VLLMVideoClient(
            base_url=str(cfg.runtime.base_url),
            api_key=str(cfg.runtime.api_key),
            model_name=str(cfg.model.served_name),
            timeout_seconds=float(cfg.runtime.timeout_seconds),
        )

    prediction_rows: list[dict[str, Any]] = []
    video_cache: dict[Path, VideoSample | Exception] = {}
    run_started_at = time.monotonic()
    total_predictions = len(annotations) * len(seeds)
    with tqdm(
        total=total_predictions,
        desc=f"{cfg.dataset.name} predictions",
        unit="pred",
        dynamic_ncols=True,
    ) as progress:
        for annotation in annotations:
            if annotation.video_path not in video_cache:
                try:
                    video_cache[annotation.video_path] = sampler(
                        annotation.video_path,
                        fps,
                        max_frames,
                    )
                except Exception as exc:
                    video_cache[annotation.video_path] = exc

            video_sample = video_cache[annotation.video_path]
            if isinstance(video_sample, Exception):
                for seed in seeds:
                    progress.set_postfix_str(
                        f"video={annotation.video_name[:24]} seed={seed}",
                        refresh=False,
                    )
                    prediction_rows.append(
                        _result_record(
                            annotation,
                            seed,
                            cfg,
                            status="video_error",
                            error_message=str(video_sample),
                        )
                    )
                    progress.update()
                continue

            for seed in seeds:
                progress.set_postfix_str(
                    f"video={annotation.video_name[:24]} seed={seed}",
                    refresh=False,
                )
                try:
                    raw_response = _generate_with_retries(
                        client,
                        video=video_sample,
                        prompt=prompts[annotation.annotation_id],
                        seed=seed,
                        temperature=temperature,
                        max_tokens=int(cfg.generation.max_tokens),
                        max_retries=max_retries,
                        retry_backoff_seconds=retry_backoff_seconds,
                    )
                except Exception as exc:
                    logger.error(
                        "vLLM request failed for %s (seed=%s); continuing: %s",
                        annotation.video_name,
                        seed,
                        exc,
                    )
                    prediction_rows.append(
                        _result_record(
                            annotation,
                            seed,
                            cfg,
                            status="inference_error",
                            error_message=str(exc),
                            sampled_frame_count=video_sample.sampled_frame_count,
                            candidate_frame_count=video_sample.candidate_frame_count,
                            sampling_mode=video_sample.sampling_mode,
                        )
                    )
                    progress.update()
                    continue

                parsed_response = parse_response(raw_response)
                prediction = parsed_response.count if parsed_response is not None else None
                row = _result_record(
                    annotation,
                    seed,
                    cfg,
                    raw_response=raw_response,
                    status="ok" if prediction is not None else "parse_error",
                    error_message=(
                        "Expected a JSON object with count, action_description, and reasoning"
                        if parsed_response is None
                        else ""
                    ),
                    sampled_frame_count=video_sample.sampled_frame_count,
                    candidate_frame_count=video_sample.candidate_frame_count,
                    sampling_mode=video_sample.sampling_mode,
                )
                row["pred_count"] = prediction
                if parsed_response is not None:
                    row["action_description"] = parsed_response.action_description
                    row["reasoning"] = parsed_response.reasoning
                if prediction is not None:
                    signed_error = prediction - annotation.gt_count
                    row["signed_error"] = signed_error
                    row["abs_error"] = abs(signed_error)
                    row["relative_abs_error"] = (
                        abs(signed_error) / annotation.gt_count
                        if annotation.gt_count > 0
                        else None
                    )
                prediction_rows.append(row)
                progress.update()

    run_status = (
        "complete_with_errors"
        if any(row["status"] != "ok" for row in prediction_rows)
        else "complete"
    )
    _write_outputs(output_dir, prediction_rows, annotations, seeds, cfg, run_status)
    print(f"Selected {split} annotations: {annotation_csv}")
    print(f"Wrote predictions and per-seed metrics to: {output_dir}")
    status_counts = {
        status: sum(row["status"] == status for row in prediction_rows)
        for status in ("ok", "parse_error", "inference_error", "video_error")
    }
    elapsed_seconds = time.monotonic() - run_started_at
    print(
        "Prediction summary: "
        f"{len(prediction_rows)}/{total_predictions} rows in {elapsed_seconds:.1f}s; "
        + ", ".join(f"{status}={count}" for status, count in status_counts.items())
    )
    if run_status == "complete_with_errors":
        print("Inspect status and error_message in predictions.csv for failed rows.")
    return output_dir, split