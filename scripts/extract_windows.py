"""Extract only the PORTIA windows you need from the 75 GB split ZIP.

1) Choose windows (writes a list file):
     python scripts/extract_windows.py --passes 6          # ~6 good ship passes (tug, cargo, passenger...)
     python scripts/extract_windows.py --task binary --n-per-split 1500   # balanced baseline subset
     python scripts/extract_windows.py --passes 6 --task binary --n-per-split 1500  # both
2) Extract them:
     python scripts/extract_windows.py ... --archive /mnt/big/PORTIA_audio.zip
   Uses 7-Zip (sudo apt install p7zip-full). 7-Zip reads the split parts (.z01-.z06) sitting next to
   PORTIA_audio.zip. If you only downloaded PORTIA_audio.zip, 7-Zip extracts whatever is inside it
   and reports the rest as missing; the notebook works with whatever was extracted.

Windows go to data/portia/audio/. The list is saved to data/portia/selected_windows.txt.
"""
import argparse, subprocess, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import portia_tools as pt

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/portia")
ap.add_argument("--passes", type=int, default=0, help="number of ship passes to take (spread over classes)")
ap.add_argument("--task", choices=list(pt.TASKS), help="also take a subset of this benchmark")
ap.add_argument("--n-per-split", type=int, default=1000, help="windows per split (train/val/test) for --task")
ap.add_argument("--archive", help="path to PORTIA_audio.zip (with .z01-.z06 beside it) to extract now")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

df = pt.load_manifests(a.root)
ids = set()
if a.passes:
    passes = pt.find_passes(df)
    classes = [c for c in pt.CLASSES if c in set(passes["cls"])]
    per = max(1, -(-a.passes // max(1, len(classes))))
    chosen = pt.pick_passes(passes, classes=classes, per_class=per).head(a.passes)
    for r in chosen.itertuples(): ids.update(r.window_ids)
    print(f"{len(chosen)} passes:", ", ".join(f"{r.cls} {r.mmsi} ({r.n_windows} win, min {r.min_dist_km:.2f} km)" for r in chosen.itertuples()))
if a.task:
    t = df[df["task"] == a.task]
    lab = "label_vessel" if a.task == "binary" else "primary_class"
    for split, g in t.groupby("split"):
        # keep the class balance of each split, cap the size
        k = min(len(g), a.n_per_split)
        s = g.groupby(lab, group_keys=False).apply(lambda q: q.sample(max(1, round(k * len(q) / len(g))), random_state=a.seed))
        ids.update(s["window_id"]); print(f"{a.task}/{split}: {len(s)} windows")
if not ids:
    sys.exit("Nothing selected: use --passes and/or --task")

names = sorted(pt.window_path(i, ".").name for i in ids)     # bare file names, matched anywhere in the archive
lst = Path(a.root) / "selected_windows.txt"
lst.write_text("\n".join(names) + "\n")
print(f"{len(names)} windows -> {lst}")

if a.archive:
    exe = shutil.which("7z") or shutil.which("7za")
    if not exe: sys.exit("7-Zip not found: sudo apt install p7zip-full")
    out = Path(a.root) / "audio"; out.mkdir(parents=True, exist_ok=True)
    # e: extract without folder paths ; -ir@list: listed names, matched in any folder ;
    # -aos: skip files already extracted ; -y: yes to all
    r = subprocess.run([exe, "e", a.archive, f"-o{out}", f"-ir@{lst}", "-aos", "-y"])
    have = sum(pt.window_path(i, out).exists() for i in ids)
    print(f"extracted {have}/{len(ids)} windows into {out}" + ("" if r.returncode == 0 else
          "  (7-Zip reported errors: probably parts of the split archive are missing)"))
