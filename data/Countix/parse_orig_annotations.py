#!/usr/bin/env python3
"""Filter Countix CSV annotations based on successfully downloaded/extracted videos.

Reads the original CSVs from the 'orig_anns' directory, checks if the video file
exists in the 'full_clip_videos/' directory, and creates new CSVs in the 'full_clip_annotations/'
directory replacing the video_id with the actual downloaded video filename.
"""

import csv
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")


def time_token(value: float) -> str:
    """Reconstruct the time token exactly as in previous scripts."""
    if value.is_integer():
        return f"{int(value):06d}"
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def main() -> int:
    videos_dir = SCRIPT_DIR / "full_clip_videos"
    orig_anns_dir = SCRIPT_DIR / "orig_anns"
    annotations_dir = SCRIPT_DIR / "full_clip_annotations"
    
    if not videos_dir.is_dir():
        print(f"Error: Directory {videos_dir.name} does not exist.", file=sys.stderr)
        return 1
        
    if not orig_anns_dir.is_dir():
        print(f"Error: Original annotations directory '{orig_anns_dir.name}' does not exist.", file=sys.stderr)
        return 1

    # Create the annotations directory if it does not exist
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Map the videos actually present on disk (stem -> full filename)
    available_videos = {}
    for video_path in videos_dir.iterdir():
        if video_path.is_file():
            available_videos[video_path.stem] = video_path.name

    print(f"Found {len(available_videos)} videos in the '{videos_dir.name}/' directory.")

    for split in SPLITS:
        input_filename = f"Countix_{split}.csv"
        input_csv = orig_anns_dir / input_filename
        
        # Fallback in case the files are named CountixAV_train.csv etc.
        if not input_csv.is_file():
            input_filename = f"CountixAV_{split}.csv"
            input_csv = orig_anns_dir / input_filename
            
        if not input_csv.is_file():
            print(f"Warning: CSV file for split '{split}' not found in '{orig_anns_dir.name}'. Skipped.")
            continue

        output_csv = annotations_dir / input_csv.name
        kept_rows = 0
        total_rows = 0

        with input_csv.open("r", encoding="utf-8-sig", newline="") as infile, \
             output_csv.open("w", encoding="utf-8", newline="") as outfile:
            
            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                print(f"Error: File {input_csv.name} is empty or malformed.")
                continue
                
            # Prepare the new headers: replace 'video_id' with 'video_name'
            fieldnames = list(reader.fieldnames)
            if "video_id" in fieldnames:
                idx = fieldnames.index("video_id")
                fieldnames[idx] = "video_name"
            
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                video_id = row.get("video_id")
                if not video_id:
                    continue
                
                total_rows += 1
                try:
                    start = float(row["kinetics_start"])
                    end = float(row["kinetics_end"])
                except (ValueError, KeyError):
                    continue
                
                # Reconstruct the expected stem
                stem = f"{video_id}_{time_token(start)}_{time_token(end)}"
                
                # If the video is present in the directory, write the row
                if stem in available_videos:
                    new_row = row.copy()
                    # Remove the old key and insert the new one with the filename
                    del new_row["video_id"]
                    new_row["video_name"] = available_videos[stem]
                    
                    writer.writerow(new_row)
                    kept_rows += 1

        print(f"[{split}] {input_csv.name} -> {output_csv.name}: saved {kept_rows}/{total_rows} valid rows.")

    return 0


if __name__ == "__main__":
    sys.exit(main())