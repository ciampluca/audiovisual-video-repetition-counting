# Zero-Shot Video Repetition Benchmark

This benchmark evaluates video-capable LVLMs on the Countix, RepCount, UCFRep,
OVR-Kinetics, and OVR-Ego4d annotations. It does not train or fine-tune models.
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

## Run

From the repository root, the default run uses Countix test annotations, the
default Qwen3-VL-8B model, the default prompt, 1 sampled frame per second,
temperature 0.2, and five seeds:

```bash
video-repetition-benchmark
```

Select another dataset, model, or prompt with Hydra config groups:

```bash
video-repetition-benchmark dataset=ucfrep model=internvl3_9b prompt=count_complete_cycles
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
prediction. Localization is disabled by default; count-only runs preserve their
existing output path, while localized runs use a distinct prompt ID such as
`count-repetitions+global-sequence`.

Run a dataset/prompt/FPS sweep against the model currently loaded by vLLM:

```bash
video-repetition-benchmark --multirun \
  dataset=countix,repcount,ucfrep,ovr_kinetics,ovr_ego4d \
  prompt=count_repetitions,count_complete_cycles,count_full_video,count_class_action,count_description_action \
  sampling.fps=0.5,1,2
```

For a model other than the default, start its server with the corresponding
`--served-model-name` and set the matching Hydra model choice. Configure seeds,
temperature (including `0`), frame cap, endpoint, retry count, and output root
through Hydra overrides, for example
`benchmark.seeds='[13,17,23,42,99]' generation.temperature=0 sampling.max_frames=128`.
Inference requests run concurrently, up to `runtime.max_concurrent_requests`
(default: 2). Lower this value if GPU or system memory is constrained, or set it
to `1` to run requests sequentially.

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

Outputs omit split directories; split is recorded in both files:

```text
results/{dataset}/{model}/video-only/{prompt}/fps-{fps}/temperature-{temperature}/
  predictions.csv
  metrics_by_seed.csv
```

`predictions.csv` contains one record per annotation row and seed, including the
source `class_name` and `description` when present, raw response, parsed count,
action description, concise motion reasoning, signed/absolute/relative error,
FPS, temperature, status, and separate `prompt_id` and `localization_id` fields.
When `localization=global_sequence`, it also includes GT and predicted sequence
intervals, normalized predictions, temporal IoU, and start/end boundary errors
in seconds.

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

The count prompts return exactly one JSON object with `count` (a non-negative
integer), `action_description` (a short description), and `reasoning` (one
concise sentence grounded in visible motion). The localization prompt additionally
returns `sequence_start_fraction` and `sequence_end_fraction` in `[0, 1]`. Invalid
JSON or fields are retained with a parse-error status and excluded from the
corresponding aggregate metrics; the raw response remains in `predictions.csv`.
Missing or unreadable videos are retained as error rows.
Transient vLLM failures (connection/timeouts, HTTP 429, and HTTP 5xx) are retried
using `runtime.max_retries` and exponential backoff. After retries are exhausted,
the failed row is recorded and the benchmark continues. `run_status` is
`complete_with_errors` if any prediction row is not `ok`; inspect `status` and
`error_message` in `predictions.csv` for details.

`generation.max_tokens` limits generated output tokens, not video frames. The
default of 128 leaves room for the structured response and can be adjusted with
a Hydra override if needed.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```