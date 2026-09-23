#!/usr/bin/env python3
"""Build a local Countix video set by cross-referencing new and old CSVs.

Place this script in the Countix project root:

    Countix/
    ├── orig_anns/         # Countix_new_train.csv, Countix_train.csv, ecc.
    ├── kinetics_clips/    # searched recursively
    ├── videos/            # created and populated as videos/
    ├── missing_videos.csv
    ├── videos_not_present_orig_annotations.csv
    └── get_countix.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_VERSION = "2.4"
SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
MEDIA_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}

NEW_REQUIRED_COLUMNS = ("video_id", "counts", "num_frames")
OLD_REQUIRED_COLUMNS = ("video_id", "kinetics_start", "kinetics_end")

MISSING_VIDEOS_COLUMNS = (
    "split",
    "new_filename",
    "youtube_id",
    "kinetics_filename",
    "source_path",
    "reason",
)

NOT_IN_ORIG_COLUMNS = (
    "split",
    "new_filename",
    "youtube_id"
)

@dataclass(frozen=True)
class MediaProbe:
    duration: float | None
    video_streams: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy Countix clips cross-referencing new and old CSVs.")
    parser.add_argument("--annotations-dir", type=Path, default=SCRIPT_DIR / "orig_anns")
    parser.add_argument("--kinetics-clips-dir", type=Path, default=SCRIPT_DIR / "kinetics_clips")
    parser.add_argument("--videos-dir", type=Path, default=SCRIPT_DIR / "videos")
    parser.add_argument("--reports-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_csv_path(directory: Path, prefix: str, split: str) -> Path:
    filename = f"{prefix}_{split}.csv"
    path = directory / filename
    if not path.is_file():
        matches = [p for p in directory.glob("*.csv") if p.name.lower() == filename.lower()]
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Non trovo il file {filename} in {directory}")
    return path


def load_old_csv(csv_path: Path) -> dict[str, list[tuple[float, float]]]:
    old_data = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = {c.strip() for c in reader.fieldnames if c}
        if not set(OLD_REQUIRED_COLUMNS).issubset(columns):
            raise ValueError(f"{csv_path}: mancano le colonne richieste (video_id, kinetics_start, kinetics_end)")
        
        for row in reader:
            y_id = row["video_id"].strip()
            if not y_id:
                continue
            try:
                k_start = float(row["kinetics_start"])
                k_end = float(row["kinetics_end"])
                old_data.setdefault(y_id, []).append((k_start, k_end))
            except ValueError:
                continue
    return old_data


def load_new_csv(csv_path: Path) -> list[str]:
    new_data = []
    seen = set()  # Usiamo un set per tenere traccia dei duplicati e ignorarli
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = {c.strip() for c in reader.fieldnames if c}
        if not set(NEW_REQUIRED_COLUMNS).issubset(columns):
            raise ValueError(f"{csv_path}: mancano le colonne richieste (video_id, counts, num_frames)")
        
        for row in reader:
            vid_id = row["video_id"].strip()
            if vid_id and vid_id not in seen:
                new_data.append(vid_id)
                seen.add(vid_id)
    return new_data


def build_kinetics_stem(youtube_id: str, k_start: float, k_end: float) -> str:
    start_str = f"{int(k_start):06d}"
    end_str = f"{int(k_end):06d}"
    return f"{youtube_id}_{start_str}_{end_str}"


def index_kinetics_clips(directory: Path) -> dict[str, Path]:
    index = {}
    print(f"Scansiono {directory.resolve()} ...", flush=True)
    count = 0
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES:
            index[path.stem] = path
            count += 1
    print(f"Scansione completata: trovati {count:,} file multimediali.")
    return index


def probe_media(path: Path, ffprobe: str) -> MediaProbe:
    cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError("ffprobe failed")
    payload = json.loads(res.stdout)
    video_streams = sum(1 for s in payload.get("streams", []) if s.get("codec_type") == "video")
    duration = payload.get("format", {}).get("duration")
    return MediaProbe(duration=float(duration) if duration else None, video_streams=video_streams)


def write_report(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise SystemExit("ffprobe non trovato nel PATH")

    kinetics_index = index_kinetics_clips(args.kinetics_clips_dir)
    args.videos_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Liste uniche per raccogliere gli errori di tutti gli split
    missing_videos_list = []
    not_in_orig_list = []
    
    copied = 0
    skipped = 0

    for split in args.splits:
        new_csv_path = get_csv_path(args.annotations_dir, "Countix_new", split)
        old_csv_path = get_csv_path(args.annotations_dir, "Countix", split)

        print(f"\nElaborazione split: {split}")
        old_data = load_old_csv(old_csv_path)
        new_filenames = load_new_csv(new_csv_path)

        for pos, new_filename in enumerate(new_filenames, 1):
            youtube_id = Path(new_filename).stem[:11]
            prefix = f"[{split}] [{pos}/{len(new_filenames)}] {new_filename}"
            
            old_entries = old_data.get(youtube_id)
            if not old_entries:
                # Se non c'è nel vecchio CSV, va SOLO in not_in_orig_list
                not_in_orig_list.append({
                    "split": split, 
                    "new_filename": new_filename, 
                    "youtube_id": youtube_id
                })
                print(f"{prefix}: ID non trovato nel vecchio CSV")
                continue

            source_path = None
            expected_stem = ""
            for k_start, k_end in old_entries:
                expected_stem = build_kinetics_stem(youtube_id, k_start, k_end)
                if expected_stem in kinetics_index:
                    source_path = kinetics_index[expected_stem]
                    break
            
            if not source_path:
                # Se è nel vecchio CSV ma manca il file fisico, va in missing_videos_list
                missing_videos_list.append({
                    "split": split, "new_filename": new_filename, "youtube_id": youtube_id, 
                    "kinetics_filename": expected_stem, "source_path": "", "reason": "Matching file not found in kinetics_clips"
                })
                print(f"{prefix}: File originale {expected_stem} non trovato in kinetics_clips")
                continue

            # SALVATAGGIO CORRETTO: usa il nome originale con gli zeri a 6 cifre invece del new_filename con i decimali
            dest_path = args.videos_dir / source_path.name

            if dest_path.is_file() and not args.overwrite:
                skipped += 1
                print(f"{prefix}: Già presente, saltato")
                continue

            try:
                probe = probe_media(source_path, ffprobe)
                if probe.video_streams == 0:
                    raise RuntimeError("Il file non ha stream video")
                
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest_path.with_suffix(".tmp")
                shutil.copy2(source_path, tmp)
                tmp.replace(dest_path)
                copied += 1
                print(f"{prefix}: Copiato da {source_path.name}")
                
            except Exception as e:
                missing_videos_list.append({
                    "split": split, "new_filename": new_filename, "youtube_id": youtube_id, 
                    "kinetics_filename": expected_stem, "source_path": str(source_path), "reason": f"Errore: {str(e)}"
                })
                print(f"{prefix}: Errore - {e}")

    # Scrittura del file unico dei mancanti (se ce ne sono)
    report_path_missing = args.reports_dir / "missing_videos.csv"
    if missing_videos_list:
        write_report(report_path_missing, MISSING_VIDEOS_COLUMNS, missing_videos_list)
    elif report_path_missing.exists():
        report_path_missing.unlink() # Rimuove file vecchio se questa volta non ci sono errori

    # Scrittura del file unico dei non presenti nel CSV originale (se ce ne sono)
    report_path_not_in_orig = args.reports_dir / "videos_not_present_orig_annotations.csv"
    if not_in_orig_list:
        write_report(report_path_not_in_orig, NOT_IN_ORIG_COLUMNS, not_in_orig_list)
    elif report_path_not_in_orig.exists():
        report_path_not_in_orig.unlink()

    print(f"\nRiepilogo Totale:")
    print(f"- Copiati: {copied}")
    print(f"- Già presenti (saltati): {skipped}")
    print(f"- File mancanti su disco: {len(missing_videos_list)}" + (f" (vedi {report_path_missing.name})" if missing_videos_list else ""))
    print(f"- File non presenti nelle annotazioni originali: {len(not_in_orig_list)}" + (f" (vedi {report_path_not_in_orig.name})" if not_in_orig_list else ""))
    
    return 0 if len(missing_videos_list) == 0 else 2

if __name__ == "__main__":
    sys.exit(main())