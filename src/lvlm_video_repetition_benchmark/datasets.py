from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Annotation:
    annotation_id: str
    annotation_row: int
    video_name: str
    video_path: Path
    gt_count: float
    annotation_split: str
    gt_sequence_start_sec: float | None = None
    gt_sequence_end_sec: float | None = None
    class_name: str | None = None
    description: str | None = None


def select_annotation_csv(dataset_dir: Path, annotation_prefix: str) -> tuple[Path, str]:
    annotations_dir = dataset_dir / "annotations"
    for split in ("test", "val"):
        csv_path = annotations_dir / f"{annotation_prefix}_{split}.csv"
        if csv_path.is_file():
            return csv_path, split
    raise FileNotFoundError(
        f"No test or validation annotation CSV found in {annotations_dir} "
        f"for prefix {annotation_prefix!r}"
    )


def load_annotations(
    dataset_dir: Path,
    dataset_slug: str,
    annotation_prefix: str,
    annotation_split: str | None = None,
) -> tuple[list[Annotation], Path, str]:
    if annotation_split is None:
        csv_path, split = select_annotation_csv(dataset_dir, annotation_prefix)
    else:
        if annotation_split not in {"train", "val", "test"}:
            raise ValueError(
                "annotation_split must be one of 'train', 'val', or 'test'"
            )
        split = annotation_split
        csv_path = dataset_dir / "annotations" / f"{annotation_prefix}_{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")
    video_dir = dataset_dir / "videos"
    annotations: list[Annotation] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"video_name", "count"}
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing required columns: "
                f"{', '.join(sorted(missing_columns))}"
            )
        interval_columns = {
            "repetition_segment_start_sec",
            "repetition_segment_end_sec",
        }
        present_interval_columns = interval_columns.intersection(reader.fieldnames or ())
        if present_interval_columns and present_interval_columns != interval_columns:
            raise ValueError(
                f"{csv_path} must contain both global repetition interval columns: "
                f"{', '.join(sorted(interval_columns))}"
            )

        for row_number, row in enumerate(reader, start=1):
            video_name = (row.get("video_name") or "").strip()
            if not video_name:
                raise ValueError(f"Empty video_name in {csv_path}, data row {row_number}")
            try:
                gt_count = float(row["count"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid count in {csv_path}, data row {row_number}: {row['count']!r}"
                ) from exc
            if not math.isfinite(gt_count) or gt_count < 0:
                raise ValueError(
                    f"Count must be finite and non-negative in {csv_path}, "
                    f"data row {row_number}: {row['count']!r}"
                )

            start_raw = (row.get("repetition_segment_start_sec") or "").strip()
            end_raw = (row.get("repetition_segment_end_sec") or "").strip()
            if bool(start_raw) != bool(end_raw):
                raise ValueError(
                    f"Incomplete repetition interval in {csv_path}, data row {row_number}"
                )
            if start_raw:
                try:
                    gt_sequence_start_sec = float(start_raw)
                    gt_sequence_end_sec = float(end_raw)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid repetition interval in {csv_path}, "
                        f"data row {row_number}"
                    ) from exc
                if (
                    not math.isfinite(gt_sequence_start_sec)
                    or not math.isfinite(gt_sequence_end_sec)
                    or gt_sequence_start_sec < 0
                    or gt_sequence_end_sec <= gt_sequence_start_sec
                ):
                    raise ValueError(
                        f"Repetition interval must be finite, non-negative, and "
                        f"increasing in {csv_path}, data row {row_number}"
                    )
            else:
                gt_sequence_start_sec = None
                gt_sequence_end_sec = None

            annotations.append(
                Annotation(
                    annotation_id=f"{dataset_slug}_{split}_{row_number:06d}",
                    annotation_row=row_number,
                    video_name=video_name,
                    video_path=video_dir / video_name,
                    gt_count=gt_count,
                    annotation_split=split,
                    gt_sequence_start_sec=gt_sequence_start_sec,
                    gt_sequence_end_sec=gt_sequence_end_sec,
                    class_name=(row.get("class") or "").strip() or None,
                    description=(row.get("description") or "").strip() or None,
                )
            )

    return annotations, csv_path, split