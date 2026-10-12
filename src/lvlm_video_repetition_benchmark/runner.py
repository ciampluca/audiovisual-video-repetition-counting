from __future__ import annotations

import csv
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Callable

from tqdm import tqdm

from lvlm_video_repetition_benchmark.datasets import Annotation, load_annotations
from lvlm_video_repetition_benchmark.metrics import compute_metrics, temporal_iou
from lvlm_video_repetition_benchmark.parsing import parse_response
from lvlm_video_repetition_benchmark.prompting import (
    PromptContextError,
    build_temporal_windows,
    compose_prompt,
    render_prompt,
)
from lvlm_video_repetition_benchmark.video import VideoSample, sample_video
from lvlm_video_repetition_benchmark.vllm_client import VLLMVideoClient


logger = logging.getLogger(__name__)


class DatasetSkippedError(RuntimeError):
    pass


def _localization_identity(cfg: Any) -> tuple[str, str]:
    localization = getattr(cfg, "localization", None)
    if bool(getattr(localization, "enabled", False)):
        return str(localization.name), str(localization.slug)
    return "Count Only", "none"


PREDICTION_COLUMNS = [
    "dataset",
    "annotation_split",
    "annotation_id",
    "annotation_row",
    "video_name",
    "class_name",
    "description",
    "video_path",
    "video_duration_sec",
    "gt_count",
    "pred_count",
    "gt_sequence_start_sec",
    "gt_sequence_end_sec",
    "pred_sequence_start_fraction",
    "pred_sequence_end_fraction",
    "pred_sequence_start_sec",
    "pred_sequence_end_sec",
    "temporal_iou",
    "sequence_start_abs_error_sec",
    "sequence_end_abs_error_sec",
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
    "reasoning_mode",
    "localization_name",
    "localization_id",
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
    "reasoning_mode",
    "localization_name",
    "localization_id",
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
    "n_localization_predictions",
    "mean_temporal_iou",
    "tiou_at_0_3_percent",
    "tiou_at_0_5_percent",
    "tiou_at_0_75_percent",
    "start_mae_sec",
    "end_mae_sec",
    "boundary_mae_sec",
]

WINDOW_PREDICTION_COLUMNS = [
    "window_size_seconds",
    "window_counts_json",
    "window_count_sum",
    "window_count_consistent",
    "window_consistency_retries",
    "window_consistency_retry_error",
]


def _prompt_identity(cfg: Any) -> tuple[str, str]:
    name = str(cfg.prompt.name)
    slug = str(cfg.prompt.slug)
    reasoning_mode = str(getattr(cfg.generation, "reasoning_mode", "none"))
    if reasoning_mode not in {"none", "step_by_step"}:
        raise ValueError("generation.reasoning_mode must be 'none' or 'step_by_step'")
    temporal_windows = getattr(cfg.generation, "temporal_windows", None)
    if bool(getattr(temporal_windows, "enabled", False)):
        window_seconds = float(getattr(temporal_windows, "window_seconds", 5.0))
        label = f"{window_seconds:g}s windows"
        name = f"{name} ({label})"
        slug = f"{slug}+windows-{window_seconds:g}s"
    if reasoning_mode == "step_by_step":
        name = f"{name} (step-by-step reasoning)"
        slug = f"{slug}+cot-step-by-step"

    configured_fields = getattr(cfg.generation, "response_fields", None)
    if isinstance(configured_fields, str):
        configured_fields = (configured_fields,)
    if configured_fields:
        effective_fields = tuple(str(field) for field in configured_fields)
    else:
        effective_fields = ("count", "action_description", "reasoning")
    if reasoning_mode == "step_by_step" and "reasoning" not in effective_fields:
        effective_fields += ("reasoning",)
    if bool(getattr(temporal_windows, "enabled", False)) and "windows" not in effective_fields:
        effective_fields += ("windows",)
    default_fields = ("count", "reasoning") if reasoning_mode == "step_by_step" else ("count",)
    if bool(getattr(temporal_windows, "enabled", False)):
        default_fields += ("windows",)
    if effective_fields != default_fields:
        fields_label = "-".join(effective_fields)
        name = f"{name} (fields: {', '.join(effective_fields)})"
        slug = f"{slug}+fields-{fields_label}"
    return name, slug


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
    video_duration_sec: float | None = None,
) -> dict[str, Any]:
    localization_name, localization_slug = _localization_identity(cfg)
    prompt_name, prompt_slug = _prompt_identity(cfg)
    return {
        "dataset": cfg.dataset.name,
        "annotation_split": annotation.annotation_split,
        "annotation_id": annotation.annotation_id,
        "annotation_row": annotation.annotation_row,
        "video_name": annotation.video_name,
        "class_name": annotation.class_name,
        "description": annotation.description,
        "video_path": str(annotation.video_path),
        "video_duration_sec": video_duration_sec,
        "gt_count": annotation.gt_count,
        "pred_count": None,
        "gt_sequence_start_sec": annotation.gt_sequence_start_sec,
        "gt_sequence_end_sec": annotation.gt_sequence_end_sec,
        "pred_sequence_start_fraction": None,
        "pred_sequence_end_fraction": None,
        "pred_sequence_start_sec": None,
        "pred_sequence_end_sec": None,
        "temporal_iou": None,
        "sequence_start_abs_error_sec": None,
        "sequence_end_abs_error_sec": None,
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
        "prompt_name": prompt_name,
        "prompt_id": prompt_slug,
        "reasoning_mode": str(getattr(cfg.generation, "reasoning_mode", "none")),
        "localization_name": localization_name,
        "localization_id": localization_slug,
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


def _generate_with_window_consistency(
    client: Any,
    *,
    video: VideoSample,
    prompt: str,
    seed: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_backoff_seconds: float,
    expected_fields: tuple[str, ...] | None,
    expected_window_count: int,
    consistency_retries: int,
) -> tuple[str, int, str]:
    windowed_max_tokens = max(max_tokens, 128 + 16 * expected_window_count)
    raw_response = _generate_with_retries(
        client,
        video=video,
        prompt=prompt,
        seed=seed,
        temperature=temperature,
        max_tokens=windowed_max_tokens,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    retry_count = 0
    retry_error = ""
    parsed = parse_response(
        raw_response,
        expected_fields=expected_fields,
        expected_window_count=expected_window_count,
    )
    while (
        parsed is not None
        and parsed.window_count_consistent is False
        and retry_count < consistency_retries
    ):
        previous_counts = json.dumps(
            {"count": parsed.count, "windows": parsed.window_counts},
            separators=(",", ":"),
        )
        correction_prompt = (
            f"{prompt}\n\nYour previous numeric answer was internally inconsistent: "
            f"the total was {parsed.count}, but the window counts summed to "
            f"{parsed.window_count_sum}. Previous answer: {previous_counts}. "
            "Re-examine the video and independently recount the whole video and every "
            "window. Return one corrected JSON object in the same format."
        )
        try:
            retry_response = _generate_with_retries(
                client,
                video=video,
                prompt=correction_prompt,
                seed=seed,
                temperature=temperature,
                max_tokens=windowed_max_tokens,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
        except Exception as exc:
            retry_error = f"{type(exc).__name__}: {exc}"
            break
        retry_count += 1
        retry_parsed = parse_response(
            retry_response,
            expected_fields=expected_fields,
            expected_window_count=expected_window_count,
        )
        if retry_parsed is None:
            retry_error = "Consistency retry returned an invalid response"
            break
        raw_response = retry_response
        parsed = retry_parsed
    return raw_response, retry_count, retry_error


def _write_outputs(
    output_dir: Path,
    prediction_rows: list[dict[str, Any]],
    annotations: list[Annotation],
    seeds: list[int],
    cfg: Any,
    run_status: str,
) -> None:
    prediction_columns = PREDICTION_COLUMNS
    if bool(getattr(getattr(cfg.generation, "temporal_windows", None), "enabled", False)):
        prediction_columns = [*PREDICTION_COLUMNS, *WINDOW_PREDICTION_COLUMNS]
    _write_csv(output_dir / "predictions.csv", prediction_columns, prediction_rows)
    metric_rows: list[dict[str, Any]] = []
    localization_name, localization_slug = _localization_identity(cfg)
    for seed in seeds:
        seed_rows = [row for row in prediction_rows if row["seed"] == seed]
        metrics = compute_metrics(seed_rows)
        metric_rows.append(
            {
                "dataset": cfg.dataset.name,
                "annotation_split": annotations[0].annotation_split if annotations else "",
                "model_name": cfg.model.name,
                "prompt_name": _prompt_identity(cfg)[0],
                "reasoning_mode": str(getattr(cfg.generation, "reasoning_mode", "none")),
                "localization_name": localization_name,
                "localization_id": localization_slug,
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
    max_concurrent_requests = int(
        getattr(runtime_cfg, "max_concurrent_requests", 1)
    )
    if max_retries < 0:
        raise ValueError("runtime.max_retries must be non-negative")
    if not math.isfinite(retry_backoff_seconds) or retry_backoff_seconds < 0:
        raise ValueError("runtime.retry_backoff_seconds must be non-negative")
    if max_concurrent_requests < 1:
        raise ValueError("runtime.max_concurrent_requests must be at least one")

    workspace_root = Path(cfg.paths.workspace_root).resolve()
    prompt_name, prompt_slug = _prompt_identity(cfg)
    temporal_windows_cfg = getattr(cfg.generation, "temporal_windows", None)
    temporal_windows_enabled = bool(getattr(temporal_windows_cfg, "enabled", False))
    reasoning_mode = str(getattr(cfg.generation, "reasoning_mode", "none"))
    max_tokens = int(cfg.generation.max_tokens)
    if reasoning_mode == "step_by_step":
        max_tokens = max(max_tokens, 256)
    window_seconds = float(getattr(temporal_windows_cfg, "window_seconds", 5.0))
    consistency_retries = int(getattr(temporal_windows_cfg, "consistency_retries", 1))
    if temporal_windows_enabled:
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("generation.temporal_windows.window_seconds must be greater than zero")
        if consistency_retries < 0:
            raise ValueError("generation.temporal_windows.consistency_retries must be non-negative")
    localization_cfg = getattr(cfg, "localization", None)
    localization_enabled = bool(getattr(localization_cfg, "enabled", False))
    _, localization_slug = _localization_identity(cfg)
    localization_instruction = str(
        getattr(localization_cfg, "instruction", "") or ""
    ).strip()
    if localization_enabled and not localization_instruction:
        raise ValueError(
            "localization.instruction must be provided when localization is enabled"
        )
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
    if localization_enabled and any(
        annotation.gt_sequence_start_sec is None
        or annotation.gt_sequence_end_sec is None
        for annotation in annotations
    ):
        raise DatasetSkippedError(
            f"Dataset {cfg.dataset.name!r} skipped for prompt {prompt_slug!r} "
            f"with localization {localization_slug!r}: "
            "global repetition interval annotations are required."
        )
    if context_fields and not annotations:
        raise DatasetSkippedError(
            f"Dataset {cfg.dataset.name!r} skipped for prompt {prompt_slug!r}: "
            f"selected annotation file {annotation_csv} contains no rows for the "
            f"required prompt context fields {', '.join(context_fields)}."
        )
    try:
        response_fields = getattr(cfg.generation, "response_fields", None)
        if isinstance(response_fields, str):
            response_fields = (response_fields,)
        if not response_fields:
            response_fields = None
        base_prompts = {
            annotation.annotation_id: render_prompt(
                str(cfg.prompt.text), annotation, context_fields
            )
            for annotation in annotations
        }
        if not temporal_windows_enabled:
            prompts = {
                annotation.annotation_id: compose_prompt(
                    base_prompts[annotation.annotation_id],
                    localization_instruction if localization_enabled else None,
                    response_fields=response_fields,
                    reasoning_mode=reasoning_mode,
                )
                for annotation in annotations
            }
    except PromptContextError as exc:
        raise DatasetSkippedError(
            f"Dataset {cfg.dataset.name!r} skipped for prompt {prompt_slug!r}: "
            f"selected annotation file {annotation_csv} has missing or invalid "
            f"required prompt context ({exc})."
        ) from exc

    output_dir = (
        Path(cfg.paths.output_root).resolve()
        / str(cfg.dataset.slug)
        / str(cfg.model.slug)
        / "video-only"
        / (
            f"{prompt_slug}+{localization_slug}"
            if localization_enabled
            else prompt_slug
        )
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

    total_predictions = len(annotations) * len(seeds)
    prediction_slots: list[dict[str, Any] | None] = [None] * total_predictions
    video_cache: dict[Path, VideoSample | Exception] = {}
    remaining_video_uses = Counter(annotation.video_path for annotation in annotations)
    run_started_at = time.monotonic()

    def record_completed(
        future: Future[Any],
        task: tuple[int, Annotation, int, VideoSample, tuple[tuple[float, float], ...]],
    ) -> None:
        result_index, annotation, seed, video_sample, window_intervals = task
        try:
            result = future.result()
            if temporal_windows_enabled:
                raw_response, consistency_retry_count, consistency_retry_error = result
            else:
                raw_response = result
                consistency_retry_count = 0
                consistency_retry_error = ""
        except Exception as exc:
            logger.error(
                "vLLM request failed for %s (seed=%s); continuing: %s",
                annotation.video_name,
                seed,
                exc,
            )
            prediction_slots[result_index] = _result_record(
                annotation,
                seed,
                cfg,
                status="inference_error",
                error_message=str(exc),
                sampled_frame_count=video_sample.sampled_frame_count,
                candidate_frame_count=video_sample.candidate_frame_count,
                sampling_mode=video_sample.sampling_mode,
                video_duration_sec=video_sample.duration,
            )
        else:
            parsed_response = parse_response(
                raw_response,
                require_localization=localization_enabled,
                expected_fields=(
                    (*response_fields, "windows")
                    if temporal_windows_enabled and response_fields is not None
                    else response_fields
                ),
                expected_window_count=(len(window_intervals) if temporal_windows_enabled else None),
            )
            prediction = parsed_response.count if parsed_response is not None else None
            localization_missing = (
                localization_enabled
                and parsed_response is not None
                and parsed_response.localization_error is not None
            )
            status = (
                "parse_error"
                if parsed_response is None
                else "localization_parse_error"
                if localization_missing
                else "ok"
            )
            row = _result_record(
                annotation,
                seed,
                cfg,
                raw_response=raw_response,
                status=status,
                error_message=(
                    parsed_response.localization_error
                    if localization_missing
                    else "Expected a JSON object matching generation.response_fields"
                    if parsed_response is None
                    else ""
                ),
                sampled_frame_count=video_sample.sampled_frame_count,
                candidate_frame_count=video_sample.candidate_frame_count,
                sampling_mode=video_sample.sampling_mode,
                video_duration_sec=video_sample.duration,
            )
            row["pred_count"] = prediction
            if parsed_response is not None:
                row["action_description"] = parsed_response.action_description
                row["reasoning"] = parsed_response.reasoning
                if (
                    localization_enabled
                    and parsed_response.sequence_start_fraction is not None
                    and parsed_response.sequence_end_fraction is not None
                ):
                    row["pred_sequence_start_fraction"] = (
                        parsed_response.sequence_start_fraction
                    )
                    row["pred_sequence_end_fraction"] = (
                        parsed_response.sequence_end_fraction
                    )
                    predicted_start = (
                        parsed_response.sequence_start_fraction * video_sample.duration
                    )
                    predicted_end = (
                        parsed_response.sequence_end_fraction * video_sample.duration
                    )
                    row["pred_sequence_start_sec"] = predicted_start
                    row["pred_sequence_end_sec"] = predicted_end
                    row["sequence_start_abs_error_sec"] = abs(
                        predicted_start - annotation.gt_sequence_start_sec
                    )
                    row["sequence_end_abs_error_sec"] = abs(
                        predicted_end - annotation.gt_sequence_end_sec
                    )
                    row["temporal_iou"] = temporal_iou(
                        annotation.gt_sequence_start_sec,
                        annotation.gt_sequence_end_sec,
                        predicted_start,
                        predicted_end,
                    )
                if temporal_windows_enabled:
                    window_counts = parsed_response.window_counts
                    row.update(
                        {
                            "window_size_seconds": window_seconds,
                            "window_counts_json": (
                                json.dumps(
                                    [
                                        {
                                            "start_sec": start,
                                            "end_sec": end,
                                            "count": count,
                                        }
                                        for (start, end), count in zip(
                                            window_intervals, window_counts or ()
                                        )
                                    ],
                                    separators=(",", ":"),
                                )
                                if window_counts is not None
                                else None
                            ),
                            "window_count_sum": parsed_response.window_count_sum,
                            "window_count_consistent": parsed_response.window_count_consistent,
                            "window_consistency_retries": consistency_retry_count,
                            "window_consistency_retry_error": consistency_retry_error,
                        }
                    )
                    if parsed_response.window_count_consistent is False:
                        logger.warning(
                            "Window counts remain inconsistent for %s (seed=%s): total=%s, windows_sum=%s",
                            annotation.video_name,
                            seed,
                            parsed_response.count,
                            parsed_response.window_count_sum,
                        )
            if prediction is not None:
                signed_error = prediction - annotation.gt_count
                row["signed_error"] = signed_error
                row["abs_error"] = abs(signed_error)
                row["relative_abs_error"] = (
                    abs(signed_error) / annotation.gt_count
                    if annotation.gt_count > 0
                    else None
                )
            prediction_slots[result_index] = row
        progress.update()

    with tqdm(
        total=total_predictions,
        desc=f"{cfg.dataset.name} predictions",
        unit="pred",
        dynamic_ncols=True,
    ) as progress:
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            pending: dict[
                Future[Any],
                tuple[int, Annotation, int, VideoSample, tuple[tuple[float, float], ...]],
            ] = {}

            def drain_completed() -> None:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    task = pending.pop(future)
                    record_completed(future, task)

            for annotation_index, annotation in enumerate(annotations):
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
                    for seed_index, seed in enumerate(seeds):
                        result_index = annotation_index * len(seeds) + seed_index
                        prediction_slots[result_index] = _result_record(
                            annotation,
                            seed,
                            cfg,
                            status="video_error",
                            error_message=str(video_sample),
                        )
                        progress.update()
                else:
                    window_intervals = (
                        build_temporal_windows(video_sample.duration, window_seconds)
                        if temporal_windows_enabled
                        else ()
                    )
                    if temporal_windows_enabled:
                        request_prompt = compose_prompt(
                            base_prompts[annotation.annotation_id],
                            localization_instruction if localization_enabled else None,
                            response_fields=response_fields,
                            window_intervals=window_intervals,
                            reasoning_mode=reasoning_mode,
                        )
                        expected_fields = (
                            (*response_fields, "windows")
                            if response_fields is not None
                            else None
                        )
                    else:
                        request_prompt = prompts[annotation.annotation_id]
                        expected_fields = None
                    for seed_index, seed in enumerate(seeds):
                        while len(pending) >= max_concurrent_requests:
                            drain_completed()
                        result_index = annotation_index * len(seeds) + seed_index
                        if temporal_windows_enabled:
                            future = executor.submit(
                                _generate_with_window_consistency,
                                client,
                                video=video_sample,
                                prompt=request_prompt,
                                seed=seed,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                max_retries=max_retries,
                                retry_backoff_seconds=retry_backoff_seconds,
                                expected_fields=expected_fields,
                                expected_window_count=len(window_intervals),
                                consistency_retries=consistency_retries,
                            )
                        else:
                            future = executor.submit(
                                _generate_with_retries,
                                client,
                                video=video_sample,
                                prompt=request_prompt,
                                seed=seed,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                max_retries=max_retries,
                                retry_backoff_seconds=retry_backoff_seconds,
                            )
                        pending[future] = (
                            result_index,
                            annotation,
                            seed,
                            video_sample,
                            window_intervals,
                        )

                remaining_video_uses[annotation.video_path] -= 1
                if remaining_video_uses[annotation.video_path] == 0:
                    del video_cache[annotation.video_path]

            while pending:
                drain_completed()

    prediction_rows = [row for row in prediction_slots if row is not None]

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
        for status in (
            "ok",
            "parse_error",
            "localization_parse_error",
            "inference_error",
            "video_error",
        )
    }
    status_summary = [
        f"{status}={status_counts[status]}"
        for status in ("ok", "parse_error", "inference_error", "video_error")
    ]
    if status_counts["localization_parse_error"]:
        status_summary.insert(
            2,
            f"localization_parse_error={status_counts['localization_parse_error']}",
        )
    elapsed_seconds = time.monotonic() - run_started_at
    print(
        "Prediction summary: "
        f"{len(prediction_rows)}/{total_predictions} rows in {elapsed_seconds:.1f}s; "
        + ", ".join(status_summary)
    )
    if run_status == "complete_with_errors":
        print("Inspect status and error_message in predictions.csv for failed rows.")
    return output_dir, split