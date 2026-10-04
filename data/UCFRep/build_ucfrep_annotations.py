import os
import csv
import json
import subprocess
import shutil
import re
from tqdm import tqdm

# --- CONFIGURATION ---
# Calculate the absolute path of the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ORIG_ANNS_DIR = os.path.join(SCRIPT_DIR, 'orig_anns')
VIDEOS_DIR = os.path.join(SCRIPT_DIR, 'videos')
OUT_ANNS_DIR = os.path.join(SCRIPT_DIR, 'annotations')
REJECTED_CSV = os.path.join(SCRIPT_DIR, 'rejected_rows.csv')

FILES_TO_PROCESS = ['ucfrep_train.csv', 'ucfrep_val.csv']

def split_camel_case(text):
    """
    Splits CamelCase or PascalCase strings into space-separated words.
    Example: 'PlayingViolin' -> 'Playing Violin'
    """
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', text)

def get_video_metadata(video_path, ffprobe_executable):
    """
    Extracts FPS, total frames, and duration using ffprobe.
    """
    cmd = [
        ffprobe_executable,
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-select_streams', 'v:0',
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        if 'streams' not in data or not data['streams']:
            return 0.0, 0, 0.0
            
        stream = data['streams'][0]
        
        # Parse FPS
        fps_str = stream.get('r_frame_rate', '0/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 0.0
        else:
            fps = float(fps_str)
            
        # Parse duration and total frames
        total_frames = int(stream.get('nb_frames', 0))
        duration = float(stream.get('duration', 0.0))
        
        # Fallbacks in case metadata is missing one of the metrics
        if total_frames == 0 and duration > 0 and fps > 0:
            total_frames = int(duration * fps)
        elif duration == 0 and total_frames > 0 and fps > 0:
            duration = total_frames / fps
            
        return fps, total_frames, duration
    except Exception:
        return 0.0, 0, 0.0

def process_datasets():
    # Check if ffprobe is installed and available in PATH
    ffprobe_executable = shutil.which('ffprobe')
    if ffprobe_executable is None:
        print("Error: 'ffprobe' not found in PATH. Please ensure FFmpeg is installed.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(OUT_ANNS_DIR, exist_ok=True)
    
    all_rejected_rows = []

    for filename in FILES_TO_PROCESS:
        orig_filepath = os.path.join(ORIG_ANNS_DIR, filename)
        out_filepath = os.path.join(OUT_ANNS_DIR, filename)
        
        if not os.path.exists(orig_filepath):
            print(f"\nWarning: {orig_filepath} not found. Skipping.")
            continue

        valid_rows = []
        max_rep_count = 0 # Track the maximum number of repetitions for dynamic CSV headers

        # Read the original CSV
        with open(orig_filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"\nProcessing {filename}...")
        
        # Process each row with a progress bar
        for row in tqdm(rows, desc=f"Checking {filename}"):
            try:
                # 1. Extract video name and class
                full_name_path = row.get('name', '')
                if not full_name_path:
                    row['reject_reason'] = "Missing 'name' field"
                    all_rejected_rows.append(row)
                    continue

                name_parts = full_name_path.split('/')
                if len(name_parts) < 3:
                    row['reject_reason'] = "Malformed 'name' field"
                    all_rejected_rows.append(row)
                    continue
                
                video_filename = name_parts[-1]
                raw_class_name = name_parts[1]
                clean_class_name = split_camel_case(raw_class_name)

                # 2. Check if video exists
                video_path = os.path.join(VIDEOS_DIR, video_filename)
                if not os.path.exists(video_path):
                    row['reject_reason'] = f"Video not found in {VIDEOS_DIR}"
                    all_rejected_rows.append(row)
                    continue

                # 3. Read video metadata using ffprobe
                fps, total_frames, video_duration_sec = get_video_metadata(video_path, ffprobe_executable)

                if fps <= 0 or total_frames <= 0:
                    row['reject_reason'] = "Invalid video metadata (fps or frames <= 0) via ffprobe"
                    all_rejected_rows.append(row)
                    continue

                # 4. Extract global segment boundaries
                try:
                    seg_start_frame = float(row.get('start_frame', 0))
                    seg_end_frame = float(row.get('end_frame', 0))
                    counts = float(row.get('counts', 0))
                except ValueError:
                    row['reject_reason'] = "Non-numeric values in start_frame, end_frame or counts"
                    all_rejected_rows.append(row)
                    continue

                # Check if segment frames are logical
                if seg_start_frame > seg_end_frame:
                    row['reject_reason'] = "segment start_frame > end_frame"
                    all_rejected_rows.append(row)
                    continue
                
                if seg_end_frame > total_frames:
                    # Minor tolerance for frame count inaccuracies in some codecs
                    if seg_end_frame <= total_frames + 5: 
                        seg_end_frame = total_frames
                    else:
                        row['reject_reason'] = f"segment end_frame ({seg_end_frame}) > total video frames ({total_frames})"
                        all_rejected_rows.append(row)
                        continue

                # 5. Build the new row structure
                new_row = {
                    'video_name': video_filename,
                    'count': counts,
                    'class': clean_class_name,
                    'fps': round(fps, 3),
                    'video_start_sec': 0.0,
                    'video_end_sec': round(video_duration_sec, 3),
                    'repetition_segment_start_sec': round(seg_start_frame / fps, 3),
                    'repetition_segment_end_sec': round(seg_end_frame / fps, 3),
                    'repetition_segment_start_frame': seg_start_frame,
                    'repetition_segment_end_frame': seg_end_frame
                }

                # 6. Process L_i columns (L1 and L2 -> rep_1, L3 and L4 -> rep_2, etc.)
                rep_idx = 1
                pairs_valid = True
                last_end_frame = -1

                for i in range(1, 300, 2):
                    l_start_val = row.get(f'L{i}')
                    l_end_val = row.get(f'L{i+1}')

                    # Stop processing pairs if we hit empty columns
                    if not l_start_val or not l_end_val or str(l_start_val).strip() == '' or str(l_end_val).strip() == '':
                        break

                    try:
                        rep_start = float(l_start_val)
                        rep_end = float(l_end_val)
                    except ValueError:
                        row['reject_reason'] = f"Non-numeric value in L{i} or L{i+1}"
                        pairs_valid = False
                        break

                    # Consistency Checks
                    if rep_start > rep_end:
                        row['reject_reason'] = f"Repetition {rep_idx} start ({rep_start}) > end ({rep_end})"
                        pairs_valid = False
                        break
                    
                    if rep_start < last_end_frame:
                        row['reject_reason'] = f"Repetition {rep_idx} starts before previous repetition ended"
                        pairs_valid = False
                        break

                    if rep_end > total_frames + 5: # Small tolerance
                        row['reject_reason'] = f"Repetition {rep_idx} end ({rep_end}) > video total frames"
                        pairs_valid = False
                        break

                    # Assign repetition pairs to new row
                    new_row[f'rep_{rep_idx}_start_frame'] = rep_start
                    new_row[f'rep_{rep_idx}_end_frame'] = rep_end
                    
                    last_end_frame = rep_end
                    rep_idx += 1

                if not pairs_valid:
                    all_rejected_rows.append(row)
                    continue

                # Update the max repetitions encountered so far for dynamic header generation
                if (rep_idx - 1) > max_rep_count:
                    max_rep_count = (rep_idx - 1)

                valid_rows.append(new_row)

            except Exception as e:
                row['reject_reason'] = f"Unexpected error: {str(e)}"
                all_rejected_rows.append(row)

        # 7. Write the valid data to the new CSV
        if valid_rows:
            # Build dynamic fieldnames based on max_rep_count
            fieldnames = [
                'video_name', 'count', 'class', 'fps', 
                'video_start_sec', 'video_end_sec',
                'repetition_segment_start_sec', 'repetition_segment_end_sec',
                'repetition_segment_start_frame', 'repetition_segment_end_frame'
            ]
            for i in range(1, max_rep_count + 1):
                fieldnames.append(f'rep_{i}_start_frame')
                fieldnames.append(f'rep_{i}_end_frame')

            with open(out_filepath, mode='w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(valid_rows)
            
            print(f"Saved cleaned data to {out_filepath}. Retained {len(valid_rows)}/{len(rows)} rows.")

    # 8. Write all rejected rows into a single CSV for inspection
    if all_rejected_rows:
        # Get all keys from original CSV plus the 'reject_reason'
        rejected_fieldnames = list(all_rejected_rows[0].keys())
        if 'reject_reason' in rejected_fieldnames:
            rejected_fieldnames.remove('reject_reason')
        rejected_fieldnames = ['reject_reason'] + rejected_fieldnames # Put reason first

        with open(REJECTED_CSV, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rejected_fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_rejected_rows)
        print(f"\nGenerated {REJECTED_CSV} with {len(all_rejected_rows)} discarded rows.")
    else:
        print("\nAll rows processed successfully! No rejected rows.")

if __name__ == "__main__":
    process_datasets()