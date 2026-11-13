# %% md
# # PORTIA walkthrough: hearing ships and matching them to AIS
#
# **PORTIA** (Port of Valencia) pairs **3-second hydrophone recordings** (3 hydrophones, 32 kHz)
# with **AIS**: every window says which vessel was nearest (`primary_mmsi`), what type it is
# (`primary_class`) and **how far away it was** (`primary_dist`, km). Labels come automatically
# from AIS, so nobody had to listen to 95,000 clips.
#
# This notebook:
# 1. loads the benchmark CSVs and checks how the train/val/test split works
# 2. finds **ship passes** (consecutive windows of one MMSI) and picks e.g. a tug and a cargo ship
# 3. shows the passes on a **map** (range rings around the hydrophone; real tracks if you add AIS positions)
# 4. joins each pass into one continuous recording: **spectrogram + AIS distance + received level**, with audio players
# 5. looks at the **3 channels** (time differences between hydrophones as the ship passes)
# 6. trains a quick **baseline vessel detector** and scores it on the official test split, including detection vs distance
#
# **Data:** Fernández García et al. (2026), *Scientific Data*, doi:10.1038/s41597-026-08384-9,
# https://zenodo.org/records/21409381 (CC BY 4.0). See the README for downloading only the parts you need.
# Every section works with whatever audio windows you have extracted. Sections 1-3 need only the CSVs.

# %%
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Audio, display, HTML
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import portia_tools as pt

ROOT = Path(os.environ.get("PORTIA_ROOT", "../data/portia" if Path.cwd().name == "notebooks" else "data/portia"))
AUDIO = ROOT / "audio"
# Hydrophone position: APPROXIMATE, read off the red marker in Fig. 1 of the PORTIA paper (central basin,
# Port of Valencia). Good to a few hundred metres at best - ask the authors for the exact coordinates.
HYDRO_LAT, HYDRO_LON = 39.4400, -0.3180
print("data:", ROOT.resolve(), "| audio windows on disk:", len(list(AUDIO.glob("*.wav"))) if AUDIO.exists() else 0)

# %% md
# ## 1. The CSVs: what's in them, and why train/val/test
# Each benchmark (binary detection, single-vessel class, multi-label) has its own train/val/test CSVs.
# A window can appear in several benchmarks. **The split is by vessel (MMSI)**: the test set contains ships
# the model never heard in training, so a good score means it learned *ship sound*, not *that particular ship*.

# %%
df = pt.load_manifests(ROOT)
win = pt.unique_windows(df)
print(f"{len(df)} rows, {len(win)} unique windows, {win['primary_mmsi'].nunique()} vessels (MMSI)")
display(df.groupby(["task", "split"]).size().unstack())

# is the split really by vessel? (no MMSI in both train and test)
# Each benchmark is split on its own, so check within a task (a ship can be in binary/train and multi/test).
for task, d in df.groupby("task"):
    m = {s: set(d[d.split == s]["primary_mmsi"].dropna()) for s in ["train", "val", "test"]}
    print(f"{task:7s} vessels in train&test: {len(m['train'] & m['test'])}  train&val: {len(m['train'] & m['val'])}"
          f"  val&test: {len(m['val'] & m['test'])}")

fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
v = win[win.get("label_vessel", 1) == 1]
v["primary_class"].value_counts().plot.bar(ax=ax[0], title="windows per vessel class")
v["primary_dist"].plot.hist(bins=40, ax=ax[1], title="distance to nearest vessel (km)")
v.groupby("primary_mmsi").size().plot.hist(bins=40, ax=ax[2], title="windows per vessel")
plt.tight_layout(); plt.show()

# %% md
# ## 2. Ship passes
# A **pass** is a run of windows from the same vessel with no gap longer than 30 s. The AIS distance alone
# already shows the story: approach, closest point, departure.

# %%
passes = pt.find_passes(df)
display(passes.drop(columns="window_ids").head(15))
chosen = pt.pick_passes(passes, classes=("tug", "cargo"))     # change classes here, e.g. ("passenger", "pilot")
display(chosen.drop(columns="window_ids"))

fig, ax = plt.subplots(figsize=(12, 3.5))
for r in chosen.itertuples():
    w = win[win.window_id.isin(r.window_ids)].sort_values("start_utc")
    t = (w.start_utc - w.start_utc.iat[0]).dt.total_seconds() / 60
    ax.plot(t, w.primary_dist, "o-", ms=3, label=f"{r.cls} {r.mmsi}")
ax.set_xlabel("minutes from start of pass"); ax.set_ylabel("AIS distance (km)"); ax.invert_yaxis()
ax.legend(); ax.set_title("AIS distance during each pass (up = closer)"); plt.show()

# %% md
# ## 3. Map
# PORTIA gives **distances** to the hydrophone, not ship positions, so each window becomes a **ring**
# around the hydrophone (radius = AIS distance, colour = time). The ship was somewhere on that ring.
# The smallest ring is the closest approach.
#
# **Real tracks:** if you have AIS positions for the same time (e.g. your own receiver, or a provider), save them as
# `data/portia/ais_tracks.csv` with columns `mmsi, timestamp, lat, lon`. The map then draws the actual tracks too.
# **The hydrophone position above is a placeholder;** set the real one.

# %%
import folium
from matplotlib import cm, colors
mp = folium.Map(location=[HYDRO_LAT, HYDRO_LON], zoom_start=14, tiles="OpenStreetMap")
folium.Marker([HYDRO_LAT, HYDRO_LON], tooltip="hydrophone (approx., from paper Fig. 1)",
              icon=folium.Icon(color="black", icon="volume-up", prefix="fa")).add_to(mp)
palette = ["red", "blue", "green", "purple", "orange", "darkred"]
for j, r in enumerate(chosen.itertuples()):
    w = win[win.window_id.isin(r.window_ids)].sort_values("start_utc")
    layer = folium.FeatureGroup(name=f"{r.cls} {r.mmsi}")
    norm = colors.Normalize(0, len(w) - 1)
    for i, (d, ts) in enumerate(zip(w.primary_dist, w.start_utc)):
        if pd.isna(d): continue
        c = colors.to_hex(cm.viridis(norm(i)))
        folium.Circle([HYDRO_LAT, HYDRO_LON], radius=float(d) * 1000, color=c, weight=1, fill=False,
                      tooltip=f"{r.cls} {r.mmsi} {ts:%H:%M:%S} {d:.2f} km").add_to(layer)
    i = int(w.primary_dist.idxmin()); dmin = w.primary_dist.min()
    folium.Circle([HYDRO_LAT, HYDRO_LON], radius=float(dmin) * 1000, color=palette[j % 6], weight=4,
                  tooltip=f"{r.cls} {r.mmsi} closest {dmin:.2f} km at {win.loc[i,'start_utc']:%H:%M:%S}").add_to(layer)
    layer.add_to(mp)
trk = ROOT / "ais_tracks.csv"
if trk.exists():
    ais = pd.read_csv(trk, parse_dates=["timestamp"])
    for j, r in enumerate(chosen.itertuples()):
        a = ais[(ais.mmsi == r.mmsi) & (ais.timestamp.between(r.start.tz_localize(None) if ais.timestamp.dt.tz is None else r.start,
                                                              r.end.tz_localize(None) if ais.timestamp.dt.tz is None else r.end))]
        if len(a): folium.PolyLine(a[["lat", "lon"]].values.tolist(), color=palette[j % 6], weight=3,
                                   tooltip=f"AIS track {r.cls} {r.mmsi}").add_to(mp)
folium.LayerControl().add_to(mp)
mp.save(str(ROOT / "passes_map.html")); mp

# %% md
# ## 4. Listening to a pass: spectrogram + AIS distance + received level
# The 3 s windows of a pass are placed on one timeline (gaps = windows not in the dataset or not extracted, shown
# as silence). Expected:
# - **broadband noise and propeller tones grow** as the ship closes and fade after the closest point
# - the **received level** (dB, 20-2000 Hz) mirrors the **AIS distance**
# - tonal lines may shift slightly (**Doppler**) as the ship goes from approaching to receding

# %%
import librosa, librosa.display

def show_pass(r, fmax=4000, play_sr=16000, max_play_s=180):
    w = win[win.window_id.isin(r.window_ids)].sort_values("start_utc")
    x, sr, t0, missing = pt.join_pass(w, AUDIO)
    if x is None:
        print(f"{r.cls} {r.mmsi}: no audio extracted for this pass yet (see README / scripts/extract_windows.py)"); return None
    tw = (w.start_utc - t0).dt.total_seconds() + pt.WIN_S / 2
    tl, lvl = pt.band_level_db(x[0], sr)
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1, 2.2]})
    ax[0].plot(tw, w.primary_dist, "o-", ms=3, c="k"); ax[0].invert_yaxis(); ax[0].set_ylabel("AIS dist (km)")
    ax[0].set_title(f"{r.cls}  MMSI {r.mmsi}  {t0:%Y-%m-%d %H:%M} UTC   ({len(missing)} of {len(w)} windows missing)")
    ax[1].plot(tl, lvl, lw=0.8); ax[1].set_ylabel("level dB\n20-2000 Hz")
    S = librosa.amplitude_to_db(np.abs(librosa.stft(x[0], n_fft=4096, hop_length=1024)), ref=np.max)
    librosa.display.specshow(S, sr=sr, hop_length=1024, x_axis="s", y_axis="linear", ax=ax[2], cmap="magma", vmin=-70)
    ax[2].set_ylim(0, fmax); ax[2].set_xlabel("seconds from start of pass  (black bands = gaps between windows; cyan = AIS closest point, green = loudest minute)")
    k = w.primary_dist.values.argmin()
    # loudest minute (smoothed level) vs AIS closest point: a lag hints at AIS latency or stern-aspect noise
    lv = pd.Series(lvl).rolling(int(60 / 0.25), center=True, min_periods=20).median().values
    t_peak = tl[np.nanargmax(lv)] if np.isfinite(lv).any() else np.nan
    for a_ in ax: a_.axvline(tw.iat[k], c="c", ls="--", lw=1); a_.axvline(t_peak, c="lime", ls=":", lw=1.2)
    ax[1].plot(tl, lv, c="lime", lw=1)
    print(f"AIS closest point at {tw.iat[k]:.0f} s, loudest (1-min median) at {t_peak:.0f} s -> lag {t_peak - tw.iat[k]:+.0f} s")
    plt.tight_layout(); plt.show()
    y = librosa.resample(x[0], orig_sr=sr, target_sr=play_sr)
    print("whole pass (hydrophone 1, up to 3 min):"); display(Audio(y[: max_play_s * play_sr], rate=play_sr))
    c = int(tw.iat[k] * play_sr); far = int(tw.iat[int(np.nanargmax(w.primary_dist.values))] * play_sr)
    print(f"closest approach ({w.primary_dist.min():.2f} km), 10 s:"); display(Audio(y[max(0, c - 5 * play_sr): c + 5 * play_sr], rate=play_sr))
    print(f"farthest point ({w.primary_dist.max():.2f} km), 10 s:"); display(Audio(y[max(0, far - 5 * play_sr): far + 5 * play_sr], rate=play_sr))
    return dict(r=r, w=w, x=x, sr=sr, tw=tw)

shown = [show_pass(r) for r in chosen.itertuples()]

# %% md
# ### Received level vs distance
# Each window's level against its AIS distance. A straight-line fit on a log-distance axis gives the
# spreading loss: about **20 dB per decade** for spherical spreading, about **10** for cylindrical (shallow water).

# %%
for s in [s for s in shown if s]:
    lv = [20 * np.log10(np.sqrt(np.mean(pt.load_window(i, AUDIO)[0][0] ** 2)) + 1e-12) if pt.load_window(i, AUDIO)[0] is not None else np.nan
          for i in s["w"].window_id]
    d = s["w"].primary_dist.values; ok = ~np.isnan(lv) & (d > 0)
    if ok.sum() < 5: continue
    b = np.polyfit(np.log10(d[ok]), np.array(lv)[ok], 1)
    plt.semilogx(d[ok], np.array(lv)[ok], "o", ms=4, label=f"{s['r'].cls} {s['r'].mmsi}: {-b[0]:.1f} dB/decade")
plt.xlabel("AIS distance (km)"); plt.ylabel("window level (dB FS)"); plt.legend(); plt.grid(alpha=.3)
plt.title("received level vs distance"); plt.show()

# %% md
# ## 5. Three hydrophones: does the time difference change as the ship passes?
# With several hydrophones, a sound arrives at slightly different times at each. The **time difference
# (GCC-PHAT)** between hydrophone pairs changes as the ship moves across the array. That's the basis of
# **bearing** estimation. Turning it into an angle needs the array geometry. The paper only says the three hydrophones sit in a rigid
# triangle at the quay wall; the spacing isn't given numerically, so ask the authors for it.
# Note: raw channel 1 is unused, so `load_window` returns raw channels 0, 2, 3 as hydrophones 1, 2, 3.

# %%
def gcc_curve(a, b, sr, fmin, fmax, max_tau=0.003):
    """Band-limited GCC-PHAT. Returns lags (ms) and the correlation curve."""
    n = len(a) + len(b); A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    f = np.fft.rfftfreq(n, 1 / sr); R = A * np.conj(B)
    R = np.where((f >= fmin) & (f <= fmax), R / (np.abs(R) + 1e-12), 0)
    cc = np.fft.irfft(R, n); m = int(max_tau * sr)
    return np.arange(-m, m + 1) / sr * 1000, np.concatenate([cc[-m:], cc[:m + 1]])

BANDS = {"ship band 50-1000 Hz": (50, 1000), "clicks 3-12 kHz": (3000, 12000)}
PAIRS = [(0, 1), (1, 2), (0, 2)]
for s in [s for s in shown if s]:
    w = s["w"]; imgs = {(b, p): [] for b in BANDS for p in PAIRS}
    for wid in w.window_id:
        x, sr = pt.load_window(wid, AUDIO)
        for b, (lo, hi) in BANDS.items():
            for p in PAIRS:
                if x is None: imgs[(b, p)].append(None); continue
                lag, c = gcc_curve(x[p[0]], x[p[1]], sr, lo, hi)
                imgs[(b, p)].append(np.abs(c) / (np.abs(c).max() + 1e-12))
    fig, ax = plt.subplots(len(BANDS), len(PAIRS), figsize=(15, 6), sharex=True, sharey=True)
    for i, b in enumerate(BANDS):
        for j, p in enumerate(PAIRS):
            L = len(lag); M = np.array([c if c is not None else np.full(L, np.nan) for c in imgs[(b, p)]])
            ax[i, j].pcolormesh(s["tw"], lag, M.T, cmap="magma", shading="nearest", vmin=0, vmax=1)
            ax[i, j].set_title(f"{b}: hydrophones {p[0]+1}-{p[1]+1}", fontsize=9)
            ax[i, j].axvline(s["tw"].iat[w.primary_dist.values.argmin()], c="c", ls="--", lw=1)
        ax[i, 0].set_ylabel("delay (ms)")
    for j in range(len(PAIRS)): ax[-1, j].set_xlabel("seconds from start of pass")
    fig.suptitle(f"{s['r'].cls} {s['r'].mmsi}: GCC-PHAT delay over the pass (bright = strong arrival; cyan = AIS closest point)")
    plt.tight_layout(); plt.show()

# %% md
# **Reading this:** a moving ship should draw a line whose delay **drifts** over the pass (the bearing changes).
# A perfectly **flat** bright line that is the same for every ship is a **fixed** source next to the array
# (snapping shrimp, quay machinery, or a reflection off the quay wall), not the ship. A useful sanity check:
# delay(1-3) should equal delay(1-2) + delay(2-3) for a single source. The largest delay you ever see also
# gives a lower bound on the hydrophone spacing (spacing >= delay x ~1500 m/s).

# %% md
# ## 6. Baseline: vessel present or not (binary detection), scored on the official test split
# A deliberately simple baseline: **log-mel statistics** (mean and std of a 64-band log-mel spectrogram,
# averaged over the 3 hydrophones) and a **logistic regression**. The regularisation strength is chosen on
# **val**, and **test** is used once at the end. Only windows you have extracted are used, so with a partial
# download the scores describe that subset. Check the counts printed.
#
# Then **detection rate vs distance**: how far away does the detector still hear ships?

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

b = df[df.task == "binary"].copy()
b = b[[pt.window_path(i, AUDIO).exists() for i in b.window_id]]
print("binary windows with audio:", b.groupby(["split", "label_vessel"]).size().unstack(fill_value=0).to_dict("index"))

cache = ROOT / "logmel_cache.npz"
F = dict(np.load(cache, allow_pickle=True)["F"].item()) if cache.exists() else {}
todo = [i for i in b.window_id if i not in F]
for n, i in enumerate(todo):
    x, sr = pt.load_window(i, AUDIO); F[i] = pt.logmel_stats(x, sr)
    if n % 500 == 0: print(f"features {n}/{len(todo)}")
if todo: np.savez(cache, F=np.array(F, dtype=object))
X = {s: np.stack([F[i] for i in b[b.split == s].window_id]) for s in ["train", "val", "test"] if (b.split == s).any()}
Y = {s: b[b.split == s].label_vessel.values for s in X}

if {"train", "val", "test"} <= set(X) and len(set(Y["train"])) == 2:
    best = max([0.01, 0.1, 1, 10], key=lambda C: f1_score(Y["val"], make_pipeline(StandardScaler(),
               LogisticRegression(C=C, max_iter=3000, class_weight="balanced")).fit(X["train"], Y["train"]).predict(X["val"])))
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=3000, class_weight="balanced")).fit(X["train"], Y["train"])
    p = clf.predict_proba(X["test"])[:, 1]; yh = p > 0.5
    print(f"C={best} (chosen on val)\nTEST  accuracy {accuracy_score(Y['test'], yh):.3f}  F1 {f1_score(Y['test'], yh):.3f}  ROC-AUC {roc_auc_score(Y['test'], p):.3f}")
    print(confusion_matrix(Y["test"], yh))
    t = b[b.split == "test"].assign(p=p, hit=yh)
    pos = t[t.label_vessel == 1].dropna(subset=["primary_dist"])
    bins = [0, .1, .25, .5, 1, 2]
    rate = pos.groupby(pd.cut(pos.primary_dist, bins))["hit"].agg(["mean", "size"])
    display(rate.rename(columns={"mean": "detection rate", "size": "windows"}))
    rate["mean"].plot.bar(); plt.ylabel("detected"); plt.xlabel("AIS distance (km)"); plt.ylim(0, 1)
    plt.title("baseline detection rate vs distance (test split)"); plt.show()
else:
    print("Need extracted audio for train, val and test with both labels - see scripts/extract_windows.py --task binary")

# %% md
# ## Next steps
# - **Better features:** swap log-mel statistics for pretrained embeddings (PANNs, BEATs or CLAP; same code,
#   different `features` function) and compare on the same test split.
# - **Use the array:** add the inter-hydrophone delays as features, or estimate bearing once the geometry is known.
# - **Your own data:** the same pipeline (windows + AIS distance + MMSI-aware split) works for your hull and winch
#   hydrophones with your own AIS receiver. PORTIA is a ready-made benchmark to compare against.
