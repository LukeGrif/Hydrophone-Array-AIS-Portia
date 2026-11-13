"""Helpers for exploring the PORTIA port acoustic dataset (Port of Valencia, hydrophones + AIS).

Dataset: Fernández García et al. (2026), "PORTIA: A Large-Scale Port Acoustic Dataset with
AIS-Grounded Annotations for AI-Based Vessel Monitoring", Scientific Data,
doi:10.1038/s41597-026-08384-9 ; data: https://zenodo.org/records/21409381 (CC BY 4.0)

Expected layout (see README):
    data/portia/
        PORTIA_Binary_Detection/               train/val/test CSVs   (from the small zips)
        PORTIA_Single_Vessel_Classification/
        PORTIA_Multi_Label_Monitoring/
        audio/<window_id>.wav                  3 s, 4 channels (hydrophones = 0, 2, 3), 32 kHz (only what you extracted)

Manifest columns (per the Zenodo description): window_id, start_utc, label_vessel, n_vessels,
primary_class, primary_dist (km), primary_mmsi, per-class 0/1 columns, dist_<class>_km.
The code tolerates missing columns where it can.
"""
from pathlib import Path
import glob, os, re
import numpy as np
import pandas as pd
import soundfile as sf

CLASSES = ["tug", "pilot", "cargo", "passenger", "support", "tanker", "other"]
TASKS = {"binary": "PORTIA_Binary_Detection",
         "single": "PORTIA_Single_Vessel_Classification",
         "multi": "PORTIA_Multi_Label_Monitoring"}
WIN_S = 3.0


# ------------------------------------------------------------------ manifests
def _task_of(path):
    p = path.lower().replace("-", "_")
    if "binary" in p: return "binary"
    if "single" in p: return "single"
    if "multi" in p: return "multi"
    return None


def load_manifests(root="data/portia"):
    """All benchmark CSVs in one table with 'task' and 'split' columns (a window can appear in
    several tasks). Finds CSVs anywhere under root; the task comes from the folder/file name
    (binary / single / multi), the split from 'train' / 'val' / 'test' in the name."""
    rows, seen = [], []
    for f in sorted(glob.glob(f"{root}/**/*.csv", recursive=True)):
        rel = os.path.relpath(f, root)
        task = _task_of(rel)
        m = re.search(r"(train|val|test)", Path(f).stem.lower())
        if task is None or m is None: continue
        df = pd.read_csv(f)
        if "window_id" not in df: continue
        df["task"], df["split"] = task, m.group(1)
        rows.append(df); seen.append(rel)
    if not rows:
        csvs = glob.glob(f"{root}/**/*.csv", recursive=True)
        raise FileNotFoundError(
            f"No PORTIA benchmark CSVs found under {Path(root).resolve()}.\n"
            f"CSV files there: {csvs[:10] or 'none'}\n"
            "Unzip PORTIA_Binary_Detection.zip, PORTIA_Single_Vessel_Classification.zip and "
            "PORTIA_Multi_Label_Monitoring.zip into that folder (or set PORTIA_ROOT).")
    print(f"loaded {len(seen)} CSV files: " + ", ".join(seen))
    df = pd.concat(rows, ignore_index=True)
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True, errors="coerce")
    for c in ["primary_dist"] + [f"dist_{k}_km" for k in CLASSES]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "primary_mmsi" in df:
        df["primary_mmsi"] = pd.to_numeric(df["primary_mmsi"], errors="coerce").astype("Int64")
    return df


def unique_windows(df):
    """One row per window (drop the duplicates caused by the task overlap)."""
    return df.sort_values("task").drop_duplicates("window_id").reset_index(drop=True)


# ------------------------------------------------------------------ ship passes
def find_passes(df, max_gap_s=30.0, min_windows=5):
    """Group windows of the same primary MMSI into passes: runs of windows whose start times
    are no more than max_gap_s apart. Returns a summary table, one row per pass."""
    w = unique_windows(df).dropna(subset=["primary_mmsi", "start_utc"]).sort_values(["primary_mmsi", "start_utc"])
    out = []
    for mmsi, g in w.groupby("primary_mmsi"):
        gap = g["start_utc"].diff().dt.total_seconds().fillna(0)
        run = (gap > max_gap_s).cumsum()
        for r, p in g.groupby(run):
            if len(p) < min_windows: continue
            out.append(dict(mmsi=int(mmsi), pass_id=f"{mmsi}_{int(r)}",
                            cls=p["primary_class"].mode().iat[0] if "primary_class" in p else "?",
                            start=p["start_utc"].iat[0], end=p["start_utc"].iat[-1] + pd.Timedelta(seconds=WIN_S),
                            n_windows=len(p),
                            min_dist_km=p["primary_dist"].min() if "primary_dist" in p else np.nan,
                            max_dist_km=p["primary_dist"].max() if "primary_dist" in p else np.nan,
                            coverage=len(p) * WIN_S / max(WIN_S, (p["start_utc"].iat[-1] - p["start_utc"].iat[0]).total_seconds() + WIN_S),
                            window_ids=list(p["window_id"])))
    return pd.DataFrame(out).sort_values(["cls", "n_windows"], ascending=[True, False]).reset_index(drop=True)


def pick_passes(passes, classes=("tug", "cargo"), per_class=1, min_coverage=0.5):
    """Best pass per class: well covered (few missing windows), long, and coming close."""
    picks = []
    for c in classes:
        p = passes[(passes["cls"] == c) & (passes["coverage"] >= min_coverage)]
        if p.empty: p = passes[passes["cls"] == c]
        p = p.assign(score=p["n_windows"] / (1 + p["min_dist_km"].fillna(1)))
        picks += list(p.sort_values("score", ascending=False).head(per_class).itertuples(index=False))
    return pd.DataFrame(picks)


# ------------------------------------------------------------------ audio
def window_path(window_id, audio_dir="data/portia/audio"):
    p = Path(audio_dir) / str(window_id)
    return p if p.suffix else p.with_suffix(".wav")


# PORTIA WAVs have 4 channels: hydrophones on zero-based channels 0, 2, 3; channel 1 is the unused
# input (silent). All share one hardware clock, so the channels are sample-synchronised (paper, Methods).
HYDRO_CH = [0, 2, 3]


def hydrophones(x):
    """Keep only the hydrophone channels of a (channels, samples) array."""
    return x[HYDRO_CH] if x.shape[0] == 4 else x


def load_window(window_id, audio_dir="data/portia/audio", hydro_only=True):
    """(channels, samples) float32, or None if the window wasn't extracted.
    By default returns the 3 hydrophones only (raw channels 0, 2, 3 -> rows 0, 1, 2)."""
    p = window_path(window_id, audio_dir)
    if not p.exists(): return None, None
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    x = x.T
    return (hydrophones(x) if hydro_only else x), sr


def join_pass(windows, audio_dir="data/portia/audio"):
    """Place the 3 s windows of a pass on one continuous timeline (by start_utc).
    Gaps (windows not in the dataset / not extracted) are filled with silence and reported.
    windows: DataFrame with window_id and start_utc. Returns (audio[ch, n], sr, t0, missing)."""
    windows = windows.sort_values("start_utc")
    t0 = windows["start_utc"].iat[0]
    total = (windows["start_utc"].iat[-1] - t0).total_seconds() + WIN_S
    sr = ch = None; audio = None; missing = []
    for wid, ts in zip(windows["window_id"], windows["start_utc"]):
        x, s = load_window(wid, audio_dir)
        if x is None: missing.append(wid); continue
        if audio is None:
            sr, ch = s, x.shape[0]; audio = np.zeros((ch, int(np.ceil(total * sr))), np.float32)
        i = int(round((ts - t0).total_seconds() * sr))
        n = min(x.shape[1], audio.shape[1] - i)
        audio[:, i:i + n] = x[:, :n]
    return audio, sr, t0, missing


def band_level_db(x, sr, hop_s=0.25, fmin=20, fmax=2000):
    """Received level in a band (dB re full scale) over time: how 'loud' the ship is."""
    from scipy.signal import butter, sosfiltfilt
    y = sosfiltfilt(butter(4, [fmin, min(fmax, sr / 2 - 1)], "band", fs=sr, output="sos"), x)
    h = int(hop_s * sr)
    rms = np.sqrt(np.array([np.mean(y[i:i + h] ** 2) for i in range(0, len(y) - h + 1, h)]))
    lvl = np.where(rms > 1e-7, 20 * np.log10(np.maximum(rms, 1e-12)), np.nan)   # NaN = gap (no window)
    return np.arange(len(rms)) * hop_s + hop_s / 2, lvl


# ------------------------------------------------------------------ features for the baseline
def logmel_stats(x, sr, n_mels=64, fmax=8000):
    """Simple, robust clip features: mean + std of the log-mel spectrogram (averaged over the 3 hydrophones)."""
    import librosa
    x = hydrophones(x)
    m = np.mean([librosa.power_to_db(librosa.feature.melspectrogram(y=c, sr=sr, n_fft=2048, hop_length=512,
                                                                     n_mels=n_mels, fmax=min(fmax, sr / 2)))
                 for c in x], axis=0)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])
