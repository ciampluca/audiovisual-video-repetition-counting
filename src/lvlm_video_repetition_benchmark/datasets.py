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
) -> tuple[list[Annotation], Path, str]:
    csv_path, split = select_annotation_csv(dataset_dir, annotation_prefix)
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

            annotations.append(
                Annotation(
                    annotation_id=f"{dataset_slug}_{split}_{row_number:06d}",
                    annotation_row=row_number,
                    video_name=video_name,
                    video_path=video_dir / video_name,
                    gt_count=gt_count,
                    annotation_split=split,
                )
            )

    return annotations, csv_path, split