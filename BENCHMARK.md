# Zero-Shot Video Repetition Benchmark

This benchmark evaluates video-capable LVLMs on the Countix, RepCount, UCFRep,
OVR-Kinetics, and OVR-Ego4d annotations. The benchmark runner does not train or
fine-tune models; an opt-in Qwen3-VL LoRA trainer is documented below.
The existing prompts use sampled video and do not include annotation metadata.
The optional `count-class-action` prompt sends the row's `class` value as text
context, and `count-description-action` sends its `description` value. These
prompts use only their named context field. Annotation counts are used only for
scoring; other annotation columns are not sent to the model.

## Setup

Use Python 3.10 or newer, FFmpeg/ffprobe, and a GPU environment compatible with
the selected vLLM build. Install the benchmark client:

```bash
python -m pip install -e .
```

Install the optional results-notebook dependencies with:

```bash
python -m pip install -e ".[notebook]"
```

## Qwen3-VL LoRA Training

Use a CUDA-compatible PyTorch environment, then install the optional training
dependencies:

```bash
python -m pip install -e ".[train]"
```

The initial experiment trains one 4-bit LoRA adapter on the official Countix
`train` split using the fixed predominant-action prompt and count-only JSON
targets. It does not use validation/test examples for training or pass
annotation class, description, audio, or repetition intervals to the model.
The default uses language-backbone attention LoRA (rank 16), gradient
checkpointing, one video per micro-batch, and samples at 10 FPS up to a cap of
3000 frames. The vision tower remains frozen. It is configured for one CUDA
GPU; adjust frame count or accumulation steps to fit available memory.

Start training from the repository root:

```bash
video-repetition-train-lora
```

Run a one-step smoke test before a full run:

```bash
video-repetition-train-lora data.max_train_samples=2 training.max_steps=1 training.gradient_accumulation_steps=1 training.epochs=1 paths.output_dir=/tmp/qwen3-vl-lora-smoke
```

The final adapter and processor are saved under
`adapters/qwen3-vl-8b-countix-lora/` by default. To serve it with vLLM:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3-vl-8b \
  --enable-lora \
  --lora-modules countix-lora=adapters/qwen3-vl-8b-countix-lora
```

Compare the base model and adapter with identical seeds, temperature, and
count-only output schema. Set distinct `model.slug` values so their result
directories do not overwrite one another. For example, evaluate the fixed and
class-name prompts on Countix with:

```bash
video-repetition-benchmark dataset=countix prompt=count_repetitions model.served_name=qwen3-vl-8b generation.response_fields='[count]' generation.temperature=0
video-repetition-benchmark dataset=countix prompt=count_repetitions model.served_name=countix-lora model.slug=qwen3-vl-8b-countix-lora generation.response_fields='[count]' generation.temperature=0
video-repetition-benchmark dataset=countix prompt=count_class_action model.served_name=countix-lora model.slug=qwen3-vl-8b-countix-lora generation.response_fields='[count]' generation.temperature=0
```

For the description prompt, use a dataset whose test annotations provide valid
descriptions (for example, OVR-Kinetics) and run the same base/adapter pair with
`prompt=count_description_action`. The adapter can be loaded under the
`countix-lora` alias while the base model remains available under
`qwen3-vl-8b`.

Install vLLM separately using its [installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/).
The three configured model IDs are:

- `Qwen/Qwen3-VL-8B-Instruct` (default)
- `llava-hf/llava-onevision-qwen2-7b-ov-hf`
- `OpenGVLab/InternVL3-9B`

Start one vLLM server for the model being evaluated. For the default model:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --served-model-name qwen3-vl-8b
```

Use the matching model ID and served name for the other configured models. Keep
the server running while evaluating datasets, prompt variants, or FPS settings
for that model. Use --max-model-len 8192 to occupy less memory. 
Use --media-io-kwargs '{"video": {"num_frames": -1}}' to overcome num-frames limitations.

## Configuration

The benchmark uses Hydra configuration groups. The default choices are
`dataset=countix`, `model=qwen3_vl_8b`, `prompt=count_repetitions`, and
`localization=global_sequence`. Available choices are:

- `dataset`: `countix`, `repcount`, `ucfrep`, `ovr_kinetics`, `ovr_ego4d`.
- `model`: `qwen3_vl_8b`, `llava_onevision_7b`, `internvl3_9b`. Each profile
  defines the Hugging Face ID, result-path slug, and vLLM served name.
- `prompt`: `count_repetitions`, `count_motion_pattern`,
  `count_action_specific_repetitions`, `count_class_action`,
  `count_description_action`. Class- and description-conditioned prompts
  require a known, non-empty matching annotation value for every selected row.
- `localization`: `none` or `global_sequence`. The default predicts one
  interval covering the overall repeated-action sequence; use `none` for
  count-only output.

Override scalar and list options directly on the command line. Core options
include:

- `benchmark.seeds`: list of inference seeds (default `[13, 17, 23, 42, 99]`).
- `sampling.fps` and `sampling.max_frames`: requested sampling rate and frame
  cap (defaults `10.0` and `3000`). If the requested rate exceeds the cap, the
  sampler uniformly subsamples the sampled timeline.
- `generation.temperature`, `generation.max_tokens`, and
  `generation.response_fields`: temperature (`0.2`), token limit (`128`), and
  JSON fields. Supported response fields are `count`, `action_description`,
  and `reasoning`; `count` is required. The default is count only.
- `generation.reasoning_mode`: `none` (default, unchanged legacy prompt) or
  `step_by_step` (a structured count procedure with a concise evidence-based
  `reasoning` field). It applies across prompt choices and composes with
  temporal windows; the output field is added automatically.
- `runtime.base_url`, `runtime.api_key`, `runtime.timeout_seconds`,
  `runtime.max_retries`, `runtime.retry_backoff_seconds`, and
  `runtime.max_concurrent_requests`: OpenAI-compatible endpoint and request
  behavior. The concurrency default is **8**; lower it to limit resource use.
- `paths.workspace_root` and `paths.output_root`: project and results roots.

Enable temporal-window counting with:

```bash
video-repetition-benchmark generation.temporal_windows.enabled=true generation.temporal_windows.window_seconds=5.0
```

The mode is disabled by default. When enabled, the prompt requests an overall
count plus counts for consecutive duration-based windows (the final window may
be shorter); the runner adds the `windows` response field automatically, so it
does not belong in `generation.response_fields`. The model receives no ground
truth counts. If a valid response's window counts do not sum to its overall
count, the runner may issue one correction request by default
(`generation.temporal_windows.consistency_retries=1`). A still-inconsistent
response is retained and marked inconsistent rather than having its total
rewritten. Malformed responses are handled by the normal parse-error path.

Enable the step-by-step reasoning variant independently of temporal windows:

```bash
video-repetition-benchmark generation.reasoning_mode=step_by_step
```

It asks the model to identify the target action, define a complete cycle from
visible motion, count cycles chronologically without extrapolating, and check
the count. It adds `reasoning` to the requested JSON fields even when
`generation.response_fields=[count]`; the field is a concise model-produced
rationale, not a guarantee of faithful internal reasoning. The same option can
be combined with windows by enabling both settings. Step-by-step mode raises
the effective token budget to at least 256 (or keeps a higher configured
`generation.max_tokens`).

Other useful overrides include `generation.temperature=0`,
`runtime.max_concurrent_requests=1`, or selecting a count-only run with
`localization=none`. Hydra run directories, `hydra.run.dir` and
`hydra.job.chdir`, can also be overridden. All fields and their defaults are
annotated in the corresponding YAML files under `configs/`.

## Run

From the repository root, the default run uses Countix test annotations, the
default Qwen3-VL-8B model, the default prompt and global-sequence localization,
10 sampled frames per second, temperature 0.2, and five seeds. The request
concurrency default is 8:

```bash
video-repetition-benchmark
```

Select another dataset, model, or prompt with Hydra config groups:

```bash
video-repetition-benchmark dataset=ucfrep model=internvl3_9b prompt=count_action_specific_repetitions
```

Use the class-conditioned prompt with a dataset whose selected annotation CSV
contains a valid `class` value for every row:

```bash
video-repetition-benchmark dataset=ucfrep prompt=count_class_action
```

The class-conditioned run is skipped before inference if the selected CSV has
no `class` column or contains a blank or `unknown` class value. That prompt uses
only the `class` column as context.

Use the description-conditioned prompt with annotations that provide a valid
description for every row, such as OVR-Kinetics or OVR-Ego4d:

```bash
video-repetition-benchmark dataset=ovr_kinetics prompt=count_description_action
```

The description-conditioned run is skipped before inference if the selected
CSV has no `description` column or any row contains a blank or `unknown`
description. The description identifies the action; the count must be based on
visible video evidence, not inferred from the text.

Add global repetition-sequence localization to any counting prompt with:

```bash
video-repetition-benchmark dataset=repcount prompt=count_repetitions localization=global_sequence
```

The localization option can be combined with every counting prompt, including
class- and description-conditioned prompts. For example:

```bash
video-repetition-benchmark dataset=ovr_kinetics prompt=count_description_action localization=global_sequence
```

It predicts one interval for the full sequence, not individual-cycle boundaries.
Normalized start/end positions are converted to seconds using the video duration.
Predictions and per-seed metrics include temporal IoU and boundary errors.
Missing or malformed localization does not invalidate an otherwise valid count
prediction. Global-sequence localization is enabled by default. Select
`localization=none` for count-only output; localized runs use a distinct prompt
ID such as `count-repetitions+global-sequence`.

Run a dataset/prompt/FPS sweep against the model currently loaded by vLLM:

```bash
video-repetition-benchmark --multirun \
  dataset=countix,repcount,ucfrep,ovr_kinetics,ovr_ego4d \
  prompt=count_repetitions,count_motion_pattern,count_action_specific_repetitions,count_class_action,count_description_action \
  sampling.fps=0.5,1,2
```

For a model other than the default, start its server with the corresponding
`--served-model-name` and set the matching Hydra model choice. Configure seeds,
temperature (including `0`), frame cap, endpoint, retry count, and output root
through Hydra overrides, for example
`benchmark.seeds='[13,17,23,42,99]' generation.temperature=0 sampling.max_frames=128`.
Inference requests run concurrently, up to `runtime.max_concurrent_requests`
(default: 8). Lower this value if GPU or system memory is constrained, or set
it to `1` to run requests sequentially.

The sampler decodes frames from the video stream only (`-map 0:v:0`), never
reads or transmits audio, and sends the sampled sequence using vLLM's
`video_url` modality. It samples at the requested FPS when that produces no more
than `sampling.max_frames`. If it would exceed the cap, it warns and uniformly
selects up to that many frames from the full sampled timeline, including its
temporal endpoints. `candidate_frame_count`, `sampled_frame_count`, and
`sampling_mode` in `predictions.csv` show whether this fallback occurred. Video
rows are not cropped using annotation boundaries. Every annotation row is
evaluated, including repeated video names with different counts.

The test annotation CSV is preferred; validation is used only if no test CSV
exists. In the current workspace UCFRep therefore uses validation, and the
other four datasets use test.

## Results

Outputs omit split directories; split is recorded in both files. Localization
is enabled by default and adds `+global-sequence` to the prompt identifier;
`localization=none` keeps count-only outputs under the prompt identifier without
that suffix:

```text
results/{dataset}/{model}/video-only/{prompt}/fps-{fps}/temperature-{temperature}/
  predictions.csv
  metrics_by_seed.csv
```

`predictions.csv` contains one record per annotation row and seed, including the
source `class_name` and `description` when present, raw response, parsed count,
any requested action description or concise motion reasoning,
signed/absolute/relative error, FPS, temperature, status, and separate
`prompt_id`, `reasoning_mode`, and `localization_id` fields. Step-by-step runs
use a distinct prompt identity; window duration and non-default response-field
schemas are also included in the result identity to avoid overwrites.
When `localization=global_sequence`, it also includes GT and predicted sequence
intervals, normalized predictions, temporal IoU, and start/end boundary errors
in seconds. With temporal windows enabled, it includes per-window predictions
and the window-sum consistency result.

- `mae_percent`: the *Every Shot Counts* MAE, computed as the mean per-row
  absolute error divided by that row's positive ground-truth count, expressed
  as a percentage. Zero-count rows are excluded from this metric and reported
  via `mae_n`; a zero prediction remains valid when its ground truth is
  positive.
- `legacy_mae_count`: unnormalized mean absolute count error, in repetitions.
  This is named LegacyMAE to distinguish it from the repetition-counting MAE.
- `rmse_count`: root mean square raw-count error.
- `obz_percent`: percentage with exact counts.
- `obo_percent`: percentage within one count.
- `mean_temporal_iou`: mean intersection-over-union for valid global sequence
  intervals; `tiou_at_0_3_percent`, `tiou_at_0_5_percent`, and
  `tiou_at_0_75_percent` report the share of intervals meeting each threshold.
- `start_mae_sec`, `end_mae_sec`, and `boundary_mae_sec`: absolute boundary
  errors in seconds for valid localization predictions.

The default count response is one JSON object with `count` (a non-negative
integer). `action_description` (a short description) and `reasoning` (one
concise sentence grounded in visible motion) can be requested through
`generation.response_fields`; step-by-step mode automatically requests its
structured `reasoning` summary. The localization prompt additionally returns
`sequence_start_fraction` and `sequence_end_fraction` in `[0, 1]`. Temporal
window mode additionally requests the per-window `windows` field. Invalid JSON
or fields are retained with a parse-error status and excluded from the
corresponding aggregate metrics; the raw response remains in `predictions.csv`.
Missing or unreadable videos are retained as error rows.
Transient vLLM failures (connection/timeouts, HTTP 429, and HTTP 5xx) are retried
using `runtime.max_retries` and exponential backoff. After retries are exhausted,
the failed row is recorded and the benchmark continues. `run_status` is
`complete_with_errors` if any prediction row is not `ok`; inspect `status` and
`error_message` in `predictions.csv` for details.

`generation.max_tokens` limits generated output tokens, not video frames. Its
default is 128. Temporal-window mode raises the effective budget to accommodate
the window list; step-by-step mode uses at least 256 tokens for its structured
rationale.

## LoRA Configuration

`configs/train_lora.yaml` controls the optional Qwen3-VL trainer. Its dataset
and prompt are Hydra groups; the trainer currently requires the Qwen3-VL-8B
profile, a `train` split, and a `val` validation split. `data.sampling_fps`,
`data.max_frames`, and `data.max_train_samples` set video sampling and the
optional training-row limit. The `lora` group sets rank, alpha, dropout, and a
regex restricted to language-backbone attention projections.

The `training` group configures response fields (`count` and optionally
`action_description`), action-description context source, seed, epochs or
maximum steps, gradient accumulation, optimizer hyperparameters, logging and
checkpoint retention, precision, gradient checkpointing, 4-bit quantization,
and optional checkpoint resume. Validation generation has its own
`validation.max_new_tokens` limit. Training saves the final adapter and
processor to `paths.output_dir`, validation history and best-NMAE/best-OBO
adapters alongside it. Every field and its default is documented inline in
`configs/train_lora.yaml`.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```