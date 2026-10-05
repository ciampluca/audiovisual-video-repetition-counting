# Zero-Shot Video Repetition Benchmark

This benchmark evaluates video-capable LVLMs on the Countix, RepCount, UCFRep,
OVR-Kinetics, and OVR-Ego4d annotations. It does not train or fine-tune models.
Only video is sampled and sent to the model. Annotation counts are used only for
scoring; action classes and descriptions are not included in requests.

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

Run a dataset/prompt/FPS sweep against the model currently loaded by vLLM:

```bash
video-repetition-benchmark --multirun \
  dataset=countix,repcount,ucfrep,ovr_kinetics,ovr_ego4d \
  prompt=count_repetitions,count_complete_cycles,count_full_video \
  sampling.fps=0.5,1,2
```

For a model other than the default, start its server with the corresponding
`--served-model-name` and set the matching Hydra model choice. Configure seeds,
temperature (including `0`), frame cap, endpoint, retry count, and output root
through Hydra overrides, for example
`benchmark.seeds='[13,17,23,42,99]' generation.temperature=0 sampling.max_frames=128`.

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
raw response, parsed count, action description, concise motion reasoning,
signed/absolute/relative error, FPS, temperature, and status.

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

The model is prompted to return exactly one JSON object with `count` (a
non-negative integer), `action_description` (a short description), and
`reasoning` (one concise sentence grounded in visible motion). The object must
contain exactly those fields, with non-empty strings for the two descriptive
fields. Invalid JSON or fields are retained with `parse_error` status and
excluded from aggregate metrics; the raw response remains in `predictions.csv`.
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