"""Make a small FAKE dataset with PORTIA's file layout and CSV columns, to test the code before
the real data is downloaded. Not real data - don't use it for results.

    python tests/make_fake_portia.py --out data/fake_portia
    PORTIA_ROOT=data/fake_portia jupyter lab notebooks/portia_ais_walkthrough.ipynb
"""
import argparse, numpy as np, pandas as pd, soundfile as sf
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/fake_portia"); a = ap.parse_args()
rng = np.random.default_rng(0); SR, W = 32000, 3.0
root = Path(a.out); (root / "audio").mkdir(parents=True, exist_ok=True)
classes = ["tug", "pilot", "cargo", "passenger", "support", "tanker", "other"]
t_start = pd.Timestamp("2025-05-12 06:00:00", tz="UTC")
rows = []

def noise(n): return 0.003 * rng.standard_normal((3, n)).astype(np.float32)

def ship_audio(d_km, f_blade, n):
    t = np.arange(n) / SR
    amp = 0.02 / max(0.05, d_km)                                   # louder when close
    bb = rng.standard_normal(n) * (1 + 0.5 * np.sin(2 * np.pi * f_blade / 5 * t))   # modulated cavitation
    tone = sum(np.sin(2 * np.pi * f_blade * k * t) / k for k in range(1, 6))
    x = amp * (0.6 * bb + 0.4 * tone)
    y = np.stack([np.roll(x, d) for d in (0, 3, 7)]).astype(np.float32) + noise(n)
    return np.insert(y, 1, 0.0, axis=0)                           # 4 ch, ch 1 unused like PORTIA

k = 0
for v in range(14):                                                # 14 vessels, 1-2 passes each
    mmsi = 224000000 + v; cls = classes[v % len(classes)]
    for p in range(1 + v % 2):
        t0 = t_start + pd.Timedelta(minutes=40 * v + 15 * p)
        d0, speed = rng.uniform(0.05, 0.6), rng.uniform(2, 6)       # closest distance km, speed m/s
        f_blade = rng.uniform(8, 25)
        for i in range(int(rng.integers(40, 80))):
            if rng.random() < 0.25: continue                         # gaps, like the real data
            ts = t0 + pd.Timedelta(seconds=W * i)
            tc = (i - 30) * W
            d = float(np.hypot(d0, speed * tc / 1000))
            if d > 1.98: continue
            wid = f"w{k:06d}.wav"; k += 1
            sf.write(root / "audio" / wid, ship_audio(d, f_blade, int(SR * W)).T, SR, subtype="PCM_16")
            r = dict(window_id=wid, start_utc=ts.isoformat(), label_vessel=1, n_vessels=1,
                     primary_class=cls, primary_dist=round(d, 3), primary_mmsi=mmsi)
            for c in classes:
                r[c] = int(c == cls); r[f"dist_{c}_km"] = round(d, 3) if c == cls else ""
            rows.append(r)
for i in range(400):                                               # background windows
    wid = f"w{k:06d}.wav"; k += 1
    sf.write(root / "audio" / wid, np.insert(noise(int(SR * W)), 1, 0.0, axis=0).T, SR, subtype="PCM_16")
    r = dict(window_id=wid, start_utc=(t_start + pd.Timedelta(hours=12, seconds=7 * i)).isoformat(),
             label_vessel=0, n_vessels=0, primary_class="", primary_dist="", primary_mmsi="")
    for c in classes: r[c] = 0; r[f"dist_{c}_km"] = ""
    rows.append(r)

df = pd.DataFrame(rows)
mm = df["primary_mmsi"].replace("", np.nan).astype(float)
split = np.where(mm.isna(), rng.choice(["train", "val", "test"], len(df), p=[.6, .2, .2]),
                 np.select([mm % 5 == 0, mm % 5 == 1], ["test", "val"], "train"))   # split by vessel
for task, folder in [("binary", "PORTIA_Binary_Detection"), ("single", "PORTIA_Single_Vessel_Classification"),
                     ("multi", "PORTIA_Multi_Label_Monitoring")]:
    (root / folder).mkdir(exist_ok=True)
    sub = df if task != "single" else df[df["label_vessel"] == 1]
    for s in ["train", "val", "test"]:
        sub[split[sub.index] == s].to_csv(root / folder / f"{s}.csv", index=False)
print(f"fake PORTIA: {len(df)} windows -> {root}")
