from __future__ import annotations

import base64
import csv
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

from lvlm_video_repetition_benchmark.datasets import Annotation, load_annotations
from lvlm_video_repetition_benchmark.metrics import compute_metrics
from lvlm_video_repetition_benchmark.parsing import parse_response
from lvlm_video_repetition_benchmark.prompting import compose_prompt
from lvlm_video_repetition_benchmark.video import sample_video


logger = logging.getLogger(__name__)


def _normalize_response_fields(response_fields: Any) -> tuple[str, ...]:
    if isinstance(response_fields, str):
        response_fields = (response_fields,)
    fields = tuple(str(field) for field in response_fields)
    if not fields or "count" not in fields:
        raise ValueError("training.response_fields must include 'count'")
    if len(set(fields)) != len(fields):
        raise ValueError("training.response_fields must not contain duplicates")
    unsupported_fields = set(fields) - {"count", "action_description"}
    if unsupported_fields:
        raise ValueError(
            "Training supports only 'count' and 'action_description'; "
            "reasoning requires ground-truth annotations"
        )
    return fields


class CountTrainingDataset:
    def __init__(
        self,
        annotations: list[Annotation],
        prompt: str,
        max_train_samples: int | None = None,
        response_fields: tuple[str, ...] = ("count",),
        split_name: str = "training",
    ) -> None:
        self.response_fields = _normalize_response_fields(response_fields)
        if max_train_samples is not None and max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive when provided")
        selected_annotations = (
            annotations[:max_train_samples]
            if max_train_samples is not None
            else annotations
        )
        if not selected_annotations:
            raise ValueError(f"The selected {split_name} annotation split contains no rows")

        missing_videos = [
            str(annotation.video_path)
            for annotation in selected_annotations
            if not annotation.video_path.is_file()
        ]
        if missing_videos:
            preview = ", ".join(missing_videos[:5])
            raise FileNotFoundError(
                f"{len(missing_videos)} {split_name} videos are missing; "
                f"first paths: {preview}"
            )
        non_integer_counts = [
            annotation.annotation_id
            for annotation in selected_annotations
            if not annotation.gt_count.is_integer()
        ]
        if non_integer_counts:
            raise ValueError(
                "JSON count supervision requires integer counts; invalid rows: "
                + ", ".join(non_integer_counts[:5])
            )
        if "action_description" in self.response_fields:
            missing_descriptions = [
                annotation.annotation_id
                for annotation in selected_annotations
                if not any(
                    isinstance(value, str)
                    and value.strip()
                    and value.strip().casefold() != "unknown"
                    for value in (annotation.description, annotation.class_name)
                )
            ]
            if missing_descriptions:
                raise ValueError(
                    "action_description supervision requires an annotation "
                    "description or class; invalid rows: "
                    + ", ".join(missing_descriptions[:5])
                )

        self.annotations = selected_annotations
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        annotation = self.annotations[index]
        target: dict[str, int | str] = {}
        for field in self.response_fields:
            if field == "count":
                target[field] = int(annotation.gt_count)
            else:
                target[field] = next(
                    value.strip()
                    for value in (annotation.description, annotation.class_name)
                    if isinstance(value, str)
                    and value.strip()
                    and value.strip().casefold() != "unknown"
                )
        return {
            "video_path": str(annotation.video_path),
            "prompt": self.prompt,
            "target": json.dumps(target, separators=(",", ":")),
        }


class Qwen3VLCountCollator:
    def __init__(self, processor: Any, sampling_fps: float, max_frames: int) -> None:
        self.processor = processor
        self.sampling_fps = sampling_fps
        self.max_frames = max_frames
        video_processor = getattr(processor, "video_processor", None)
        if video_processor is not None and hasattr(video_processor, "do_sample_frames"):
            video_processor.do_sample_frames = False

    def _read_frames(self, video_path: Path) -> tuple[list[Any], Any]:
        from PIL import Image

        sample = sample_video(video_path, self.sampling_fps, self.max_frames)
        encoded_frames = sample.video_data_uri.partition(",")[2]
        frames = []
        for encoded_frame in encoded_frames.split(","):
            with Image.open(io.BytesIO(base64.b64decode(encoded_frame))) as image:
                frames.append(image.convert("RGB"))
        if not frames:
            raise ValueError(f"No video frames decoded from {video_path}")
        return frames, sample

    def _user_message(self, example: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        from transformers.video_utils import VideoMetadata

        video_frames, video_sample = self._read_frames(Path(example["video_path"]))
        video_metadata = VideoMetadata(
            total_num_frames=video_sample.total_num_frames,
            fps=video_sample.source_fps,
            duration=video_sample.duration,
            frames_indices=video_sample.frames_indices,
        )
        user_message = {
            "role": "user",
            "content": [
                {"type": "video", "video": video_frames},
                {"type": "text", "text": example["prompt"]},
            ],
        }
        return user_message, video_metadata

    def prepare_generation_inputs(self, example: dict[str, Any]) -> dict[str, Any]:
        user_message, video_metadata = self._user_message(example)
        encoded = self.processor.apply_chat_template(
            [user_message],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            video_metadata=video_metadata,
        )
        features = dict(encoded)
        features.pop("token_type_ids", None)
        return features

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if len(examples) != 1:
            raise ValueError(
                "The initial Qwen3-VL trainer supports micro-batches of size 1; "
                "use gradient_accumulation_steps to increase the effective batch size"
            )

        example = examples[0]
        user_message, video_metadata = self._user_message(example)
        messages = [
            user_message,
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["target"]}],
            },
        ]
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            return_assistant_tokens_mask=True,
            video_metadata=video_metadata,
        )
        features = dict(encoded)
        assistant_masks = features.pop("assistant_masks", None)
        features.pop("token_type_ids", None)
        input_ids = features["input_ids"]
        labels = input_ids.clone()

        if assistant_masks is not None:
            assistant_masks = torch.as_tensor(assistant_masks, device=labels.device)
            if assistant_masks.ndim == 1:
                assistant_masks = assistant_masks.unsqueeze(0)
            if torch.any(assistant_masks != 0):
                labels[assistant_masks == 0] = -100
            else:
                assistant_masks = None
        if assistant_masks is None:
            prompt_inputs = self.processor.apply_chat_template(
                [user_message],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                video_metadata=video_metadata,
            )
            prompt_ids = prompt_inputs["input_ids"]
            prompt_length = prompt_ids.shape[-1]
            if not torch.equal(input_ids[:, :prompt_length], prompt_ids):
                raise ValueError(
                    "The Qwen3-VL chat template did not preserve the user prompt "
                    "as a prefix; cannot safely mask count-only labels"
                )
            labels[:, :prompt_length] = -100

        attention_mask = features.get("attention_mask")
        if attention_mask is not None:
            labels[attention_mask == 0] = -100
        if not torch.any(labels != -100):
            raise ValueError("The assistant response produced no supervised tokens")

        features["labels"] = labels
        return features


def _validation_count_metrics(
    ground_truth_counts: list[float], predicted_counts: list[int]
) -> dict[str, float | int]:
    if not ground_truth_counts or len(ground_truth_counts) != len(predicted_counts):
        raise ValueError("Validation requires one parsed prediction per annotation")
    metrics = compute_metrics(
        [
            {"gt_count": ground_truth, "pred_count": prediction}
            for ground_truth, prediction in zip(ground_truth_counts, predicted_counts)
        ]
    )
    if metrics["nmae"] is None:
        raise ValueError("NMAE is undefined because validation mean count is zero")
    return {
        "n": int(metrics["n"]),
        "mae": float(metrics["legacy_mae_count"]),
        "nmae": float(metrics["nmae"]),
        "obo_percent": float(metrics["obo_percent"]),
    }


def train_qwen3_vl_lora(cfg: Any) -> None:
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen3VLForConditionalGeneration,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            'Training dependencies are missing. Install them with '
            '`python -m pip install -e ".[train]"`.'
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL LoRA training requires a CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "This initial trainer is configured for one CUDA GPU; launch it with "
            "a single visible device"
        )

    dataset_dir = Path(cfg.paths.workspace_root) / "data" / str(cfg.dataset.folder)
    annotations, annotation_csv, split = load_annotations(
        dataset_dir,
        str(cfg.dataset.slug),
        str(cfg.dataset.annotation_prefix),
        annotation_split=str(cfg.data.split),
    )
    if split != "train":
        raise ValueError(f"LoRA training must use the train split, got {split!r}")

    response_fields = _normalize_response_fields(cfg.training.response_fields)
    prompt = compose_prompt(
        str(cfg.prompt.text), response_fields=response_fields
    )
    train_dataset = CountTrainingDataset(
        annotations,
        prompt,
        max_train_samples=(
            int(cfg.data.max_train_samples)
            if cfg.data.max_train_samples is not None
            else None
        ),
        response_fields=response_fields,
        split_name="training",
    )
    validation_split = str(cfg.data.validation_split)
    if validation_split != "val":
        raise ValueError("LoRA model selection must use the validation split ('val')")
    validation_annotations, validation_csv, loaded_validation_split = load_annotations(
        dataset_dir,
        str(cfg.dataset.slug),
        str(cfg.dataset.annotation_prefix),
        annotation_split=validation_split,
    )
    if loaded_validation_split != "val":
        raise ValueError(
            f"Expected validation annotations, got {loaded_validation_split!r}"
        )
    validation_dataset = CountTrainingDataset(
        validation_annotations,
        prompt,
        response_fields=response_fields,
        split_name="validation",
    )
    logger.info(
        "Training on %d rows from %s; validating on %d rows from %s with fields %s",
        len(train_dataset),
        annotation_csv,
        len(validation_dataset),
        validation_csv,
        ", ".join(response_fields),
    )

    set_seed(int(cfg.training.seed))
    use_bf16 = bool(cfg.training.bf16) and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    processor = AutoProcessor.from_pretrained(str(cfg.model.hf_id))

    quantization_config = None
    model_load_kwargs: dict[str, Any] = {
        "torch_dtype": model_dtype,
        "attn_implementation": "sdpa",
    }
    if bool(cfg.training.load_in_4bit):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(cfg.training.quant_type),
            bnb_4bit_use_double_quant=bool(cfg.training.double_quant),
            bnb_4bit_compute_dtype=model_dtype,
        )
        model_load_kwargs["quantization_config"] = quantization_config
        model_load_kwargs["device_map"] = {"": torch.cuda.current_device()}

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(cfg.model.hf_id), **model_load_kwargs
    )
    model.config.use_cache = False
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
        )

    target_modules = str(cfg.lora.target_modules)
    target_pattern = re.compile(target_modules)
    matched_modules = [
        name
        for name, _ in model.named_modules()
        if target_pattern.fullmatch(name)
    ]
    if not matched_modules:
        raise RuntimeError(
            "The configured LoRA target regex matched no Qwen3-VL modules: "
            f"{target_modules!r}. Check the installed Transformers model structure."
        )
    if any("language_model.layers." not in name for name in matched_modules):
        raise RuntimeError("LoRA targets must stay within the language backbone")

    model = get_peft_model(
        model,
        LoraConfig(
            r=int(cfg.lora.rank),
            lora_alpha=int(cfg.lora.alpha),
            lora_dropout=float(cfg.lora.dropout),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable_parameters = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_parameters or any(
        "language_model.layers." not in name for name in trainable_parameters
    ):
        raise RuntimeError("Only language-backbone LoRA parameters may be trainable")
    model.print_trainable_parameters()

    output_dir = Path(str(cfg.paths.output_dir)).expanduser().resolve()
    sampling_fps = float(cfg.data.sampling_fps)
    max_frames = int(cfg.data.max_frames)
    if sampling_fps <= 0 or max_frames <= 0:
        raise ValueError("data.sampling_fps and data.max_frames must be positive")
    validation_max_new_tokens = int(cfg.validation.max_new_tokens)
    if validation_max_new_tokens <= 0:
        raise ValueError("validation.max_new_tokens must be positive")
    data_collator = Qwen3VLCountCollator(
        processor,
        sampling_fps=sampling_fps,
        max_frames=max_frames,
    )

    class GenerationValidationCallback(TrainerCallback):
        def __init__(self) -> None:
            self.history_path = output_dir / "validation_metrics.csv"
            self.best_path = output_dir / "best_metrics.json"
            self.best_records: dict[str, Any] = {"nmae": None, "obo": None}
            if cfg.training.resume_from_checkpoint and self.best_path.is_file():
                self.best_records.update(json.loads(self.best_path.read_text(encoding="utf-8")))

        def on_epoch_end(
            self,
            args: Any,
            state: Any,
            control: Any,
            model: Any,
            **kwargs: Any,
        ) -> Any:
            was_training = model.training
            model.eval()
            ground_truth_counts: list[float] = []
            predicted_counts: list[int] = []
            try:
                for index, annotation in enumerate(validation_dataset.annotations):
                    example = validation_dataset[index]
                    features = data_collator.prepare_generation_inputs(example)
                    input_length = features["input_ids"].shape[-1]
                    device = next(model.parameters()).device
                    model_inputs = {
                        key: value.to(device) if hasattr(value, "to") else value
                        for key, value in features.items()
                    }
                    with torch.inference_mode():
                        generated = model.generate(
                            **model_inputs,
                            max_new_tokens=validation_max_new_tokens,
                            do_sample=False,
                            use_cache=True,
                        )
                    response_tokens = generated[:, input_length:]
                    response = processor.batch_decode(
                        response_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0].strip()
                    parsed = parse_response(
                        response, expected_fields=response_fields
                    )
                    if parsed is None:
                        raise ValueError(
                            "Validation response for "
                            f"{annotation.annotation_id} did not match "
                            f"response_fields={response_fields}: {response[:200]!r}"
                        )
                    ground_truth_counts.append(annotation.gt_count)
                    predicted_counts.append(parsed.count)
            finally:
                if was_training:
                    model.train()

            metrics = _validation_count_metrics(
                ground_truth_counts, predicted_counts
            )
            epoch = int(round(float(state.epoch or 0)))
            history_row = {"epoch": epoch, **metrics}
            write_header = not self.history_path.is_file() or self.history_path.stat().st_size == 0
            with self.history_path.open("a", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=("epoch", "n", "mae", "nmae", "obo_percent"),
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(history_row)

            logger.info(
                "Validation epoch %d: MAE=%.4f NMAE=%.4f OBO=%.2f%% (%d videos)",
                epoch,
                metrics["mae"],
                metrics["nmae"],
                metrics["obo_percent"],
                metrics["n"],
            )
            current_nmae = self.best_records["nmae"]
            current_obo = self.best_records["obo"]
            better_nmae = current_nmae is None or metrics["nmae"] < current_nmae["nmae"]
            better_obo = (
                current_obo is None
                or metrics["obo_percent"] > current_obo["obo_percent"]
                or (
                    metrics["obo_percent"] == current_obo["obo_percent"]
                    and metrics["nmae"] < current_obo["nmae"]
                )
            )
            for metric_name, is_better in (("nmae", better_nmae), ("obo", better_obo)):
                if not is_better:
                    continue
                adapter_dir = output_dir / f"best_{metric_name}"
                model.save_pretrained(str(adapter_dir))
                processor.save_pretrained(str(adapter_dir))
                self.best_records[metric_name] = {
                    "epoch": epoch,
                    **metrics,
                    "adapter_dir": str(adapter_dir),
                }
                logger.info("Saved best-%s adapter to %s", metric_name, adapter_dir)

            temporary_path = self.best_path.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(self.best_records, indent=2), encoding="utf-8"
            )
            temporary_path.replace(self.best_path)
            return control

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(cfg.training.epochs),
        max_steps=int(cfg.training.max_steps),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
        learning_rate=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        warmup_ratio=float(cfg.training.warmup_ratio),
        logging_steps=int(cfg.training.logging_steps),
        save_strategy="epoch",
        save_total_limit=int(cfg.training.save_total_limit),
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if quantization_config is not None else "adamw_torch",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        seed=int(cfg.training.seed),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=processor,
        callbacks=[GenerationValidationCallback()],
    )
    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint or None)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info("Saved Qwen3-VL LoRA adapter and processor to %s", output_dir)