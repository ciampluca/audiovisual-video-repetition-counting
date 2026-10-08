import json
import subprocess
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Calcola la cartella in cui si trova fisicamente questo script
SCRIPT_DIR = Path(__file__).resolve().parent

# Unisce il percorso dello script alle cartelle
ANNOTATIONS_DIR = SCRIPT_DIR / "full_clip_annotations"
VIDEO_ROOT = SCRIPT_DIR / "full_clip_videos"
SPLITS = ["train", "val", "test"]

def get_video_duration(video_path: Path) -> float:
    """Estrae la durata del video in secondi usando ffprobe in modo ultra-veloce."""
    if not video_path.is_file():
        return 0.0

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path)
    ]

    try:
        # Timeout di 5 secondi per evitare blocchi su file danneggiati
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        # In caso di file non leggibile o errori di ffprobe, restituisce 0.0
        return 0.0

def process_split(split: str):
    csv_path = ANNOTATIONS_DIR / f"Countix_{split}.csv"
    
    if not csv_path.exists():
        print(f"⚠️ File non trovato: {csv_path}")
        return
        
    print(f"\nElaborazione di {csv_path.name}...")
    df = pd.read_csv(csv_path)
    
    start_secs = []
    end_secs = []
    
    # Scorre i video mostrando una barra di avanzamento nel terminale
    for video_name in tqdm(df["video_name"], desc=f"Scansione {split}"):
        video_path = VIDEO_ROOT / str(video_name)
        duration = get_video_duration(video_path)
        start_secs.append(0.0)
        end_secs.append(duration)
        
    # Inserisce le colonne nella posizione corretta
    if "fps" in df.columns:
        # Rimuove le colonne se lo script è già stato eseguito in precedenza
        if "video_start_sec" in df.columns:
            df.drop(columns=["video_start_sec", "video_end_sec"], inplace=True)
            
        fps_idx = df.columns.get_loc("fps")
        df.insert(fps_idx + 1, "video_start_sec", start_secs)
        df.insert(fps_idx + 2, "video_end_sec", end_secs)
    else:
        # Fallback nel caso in cui la colonna 'fps' non esista
        df["video_start_sec"] = start_secs
        df["video_end_sec"] = end_secs
        
    # Salva sovrascrivendo il CSV originale
    df.to_csv(csv_path, index=False)
    print(f"✅ Aggiornato con successo: {csv_path.name}")

if __name__ == "__main__":
    for split in SPLITS:
        process_split(split)
    print("\n🎉 Tutti i file CSV sono stati elaborati.")