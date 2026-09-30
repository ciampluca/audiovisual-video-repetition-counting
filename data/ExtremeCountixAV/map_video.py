#!/usr/bin/env python3
"""Generate a CSV mapping each video to its subfolder."""

import csv
from pathlib import Path

# Absolute path to the directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = SCRIPT_DIR / "videos"
OUTPUT_CSV = SCRIPT_DIR / "video_mapping.csv"

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}

def main():
    if not VIDEOS_DIR.is_dir():
        print(f"Error: Directory does not exist at path:\n{VIDEOS_DIR}")
        return

    video_count = 0
    folder_count = 0

    with OUTPUT_CSV.open(mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "folder_name"])

        for folder in VIDEOS_DIR.iterdir():
            if folder.is_dir():
                folder_count += 1
                for file_path in folder.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS:
                        writer.writerow([file_path.name, folder.name])
                        video_count += 1

    print(f"Mapping completed successfully: '{OUTPUT_CSV.name}' generated in:\n{SCRIPT_DIR}")
    print(f"Found {video_count} videos in {folder_count} folders.")

if __name__ == "__main__":
    main()