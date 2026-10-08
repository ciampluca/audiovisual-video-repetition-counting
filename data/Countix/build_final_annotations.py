#!/usr/bin/env python3
"""Modifica le annotazioni CSV aggiungendo le suddivisioni uniformi per ripetizione.

- Sposta 'class' in 3a posizione (usa 'unknown' se assente).
- Calcola FPS e lo mette in 4a posizione.
- Rinomina le colonne dei segmenti in 'repetition_segment_start/end_sec' e '_frame'.
- Genera dinamicamente le colonne rep_1_start_frame ... rep_N_end_frame 
  dividendo uniformemente il segmento.
"""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")

def get_fps(video_path: Path, ffprobe_path: str) -> float:
    """Estrae il frame rate del video usando ffprobe."""
    cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "json",
        str(video_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe ha fallito per {video_path.name}")
        
    try:
        data = json.loads(result.stdout)
        rate_str = data["streams"][0]["r_frame_rate"]
        num, den = rate_str.split('/')
        return float(num) / float(den)
    except (KeyError, IndexError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"Impossibile determinare gli FPS per {video_path.name}: {exc}")

def main() -> int:
    annotations_dir = SCRIPT_DIR / "full_clip_annotations"
    videos_dir = SCRIPT_DIR / "full_clip_videos"
    
    if not annotations_dir.is_dir():
        print(f"Errore: La cartella {annotations_dir.name} non esiste.", file=sys.stderr)
        return 1
        
    if not videos_dir.is_dir():
        print(f"Errore: La cartella {videos_dir.name} non esiste.", file=sys.stderr)
        return 1

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        print("Errore: ffprobe non trovato nel PATH.", file=sys.stderr)
        return 1

    for split in SPLITS:
        csv_file = annotations_dir / f"Countix_{split}.csv"
        if not csv_file.is_file():
            csv_file = annotations_dir / f"CountixAV_{split}.csv"
        if not csv_file.is_file():
            print(f"\n[{split.upper()}] File non trovato in {annotations_dir.name}/. Salto.")
            continue
            
        temp_csv = csv_file.with_suffix(".tmp.csv")
        error_rows = 0
        
        # Usiamo due passate: la prima per processare i dati e trovare il Max N, 
        # la seconda per scrivere il CSV in modo coerente.
        processed_data = []
        max_n = 0
        original_fields = []

        # PASSATA 1: Lettura e calcoli in memoria
        with csv_file.open("r", encoding="utf-8-sig", newline="") as infile:
            reader = csv.DictReader(infile)
            original_fields = list(reader.fieldnames or [])
            
            for row in reader:
                vid_name = row.get("video_name")
                if not vid_name:
                    continue
                    
                video_path = videos_dir / vid_name
                if not video_path.is_file():
                    print(f"Avviso: Video {vid_name} non trovato su disco. Salto la riga.")
                    error_rows += 1
                    continue
                    
                try:
                    k_start = float(row["kinetics_start"])
                    r_start = float(row["repetition_start"])
                    r_end = float(row["repetition_end"])
                    count_val = float(row.get("count", 0))
                    n_reps = int(round(count_val)) # arrotondiamo nel caso in cui count fosse float, es. 3.0
                    
                    # Calcolo tempi normalizzati
                    norm_start_sec = round(r_start - k_start, 2)
                    norm_end_sec = round(r_end - k_start, 2)
                    
                    fps_exact = get_fps(video_path, ffprobe)
                    fps_rounded = round(fps_exact, 2)
                    
                    # Calcolo frame del segmento intero
                    seg_start_frame = int(round(norm_start_sec * fps_exact))
                    seg_end_frame = int(round(norm_end_sec * fps_exact))
                    
                except Exception as exc:
                    print(f"Errore processando la riga per {vid_name}: {exc}")
                    error_rows += 1
                    continue

                # Raccogliamo i dati base elaborati
                row_data = {
                    "original_row": row,
                    "fps_rounded": fps_rounded,
                    "norm_start_sec": norm_start_sec,
                    "norm_end_sec": norm_end_sec,
                    "seg_start_frame": seg_start_frame,
                    "seg_end_frame": seg_end_frame,
                    "n_reps": n_reps,
                    "rep_frames": {}
                }
                
                # Calcolo delle singole ripetizioni equamente divise
                if n_reps > 0:
                    max_n = max(max_n, n_reps)
                    total_frames = seg_end_frame - seg_start_frame
                    
                    for i in range(1, n_reps + 1):
                        # Usiamo la proporzione esatta e poi arrotondiamo per evitare buchi/sovrapposizioni
                        start_offset = round((i - 1) * total_frames / n_reps)
                        end_offset = round(i * total_frames / n_reps)
                        
                        row_data["rep_frames"][f"rep_{i}_start_frame"] = seg_start_frame + start_offset
                        row_data["rep_frames"][f"rep_{i}_end_frame"] = seg_start_frame + end_offset
                
                processed_data.append(row_data)

        # Costruiamo l'intestazione finale
        new_fields = ["video_name", "count", "class", "fps"]
        
        columns_to_exclude = {
            "video_name", "count", "class", "fps", 
            "kinetics_start", "kinetics_end", 
            "repetition_start", "repetition_end"
        }
        
        for field in original_fields:
            if field not in columns_to_exclude:
                new_fields.append(field)
                
        new_fields.extend([
            "repetition_segment_start_sec", 
            "repetition_segment_end_sec", 
            "repetition_segment_start_frame", 
            "repetition_segment_end_frame"
        ])
        
        # Aggiungiamo le colonne per le singole ripetizioni basandoci sul max_n trovato
        for i in range(1, max_n + 1):
            new_fields.append(f"rep_{i}_start_frame")
            new_fields.append(f"rep_{i}_end_frame")

        # PASSATA 2: Scrittura su file
        with temp_csv.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=new_fields)
            writer.writeheader()
            
            for p_data in processed_data:
                orig_row = p_data["original_row"]
                new_row = {}
                
                new_row["video_name"] = orig_row.get("video_name", "")
                new_row["count"] = orig_row.get("count", "")
                # Se la classe non c'è, mette "unknown"
                new_row["class"] = orig_row.get("class", "unknown") if orig_row.get("class") else "unknown"
                new_row["fps"] = p_data["fps_rounded"]
                
                for field in original_fields:
                    if field not in columns_to_exclude:
                        new_row[field] = orig_row.get(field, "")
                        
                new_row["repetition_segment_start_sec"] = p_data["norm_start_sec"]
                new_row["repetition_segment_end_sec"] = p_data["norm_end_sec"]
                new_row["repetition_segment_start_frame"] = p_data["seg_start_frame"]
                new_row["repetition_segment_end_frame"] = p_data["seg_end_frame"]
                
                # Inseriamo i valori dei singoli frame, se mancano il CSV li lascia vuoti di default
                for key, val in p_data["rep_frames"].items():
                    new_row[key] = val
                    
                writer.writerow(new_row)

        temp_csv.replace(csv_file)
        
        print(f"\n=== SPLIT: {split.upper()} ===")
        print(f"File aggiornato: {csv_file.name}")
        print(f"Righe elaborate: {len(processed_data)}")
        print(f"Numero massimo di ripetizioni rilevato: {max_n}")
        if error_rows > 0:
            print(f"Righe saltate per errori: {error_rows}")

    return 0

if __name__ == "__main__":
    sys.exit(main())