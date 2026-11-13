# Hydrophone Array + AIS: exploring PORTIA

Tools and a walkthrough notebook for **PORTIA**, a port acoustic dataset from the **Port of Valencia**:
3 hydrophones recorded through one audio interface (3 s windows, 32 kHz), with every window linked to **AIS**
(nearest vessel's MMSI, type and distance). Labels come automatically from AIS.

> Fernández García, A., Belmonte Hernández, A., Higon, J. M., Company, R., Pérez Carrasco, F. J., Álvarez García, F.,
> & Gutiérrez, Á. (2026). *PORTIA: A Large-Scale Port Acoustic Dataset with AIS-Grounded Annotations for AI-Based
> Vessel Monitoring.* Scientific Data. https://doi.org/10.1038/s41597-026-08384-9
> Data: https://zenodo.org/records/21409381 (CC BY 4.0)

## What the notebook shows (`notebooks/portia_ais_walkthrough.ipynb`)
1. **The CSVs and the split.** The three benchmarks (binary detection, single-vessel class, multi-label) are split **by vessel (MMSI)**, so test ships are never heard in training.
2. **Ship passes.** Consecutive windows of one MMSI. Picks e.g. a tug and a cargo ship and plots their AIS distance over time.
3. **Map.** Range rings around the hydrophone (radius = AIS distance, colour = time) on OpenStreetMap, plus real AIS tracks if you add positions (`data/portia/ais_tracks.csv`: `mmsi,timestamp,lat,lon`).
4. **Spectrogram + AIS distance + received level** for each pass, with audio players (whole pass, closest approach, farthest point), and **level vs distance** (spreading loss).
5. **Three hydrophones.** Inter-hydrophone time differences (GCC-PHAT) over a pass: the basis of bearing estimation.
6. **Baseline detector.** Log-mel statistics + logistic regression for vessel present/absent, tuned on **val**, scored once on the official **test** split, and **detection rate vs distance**.

## Get the data (you don't need all 75 GB)
Zenodo record files:

| File | Size | Needed for |
|---|---|---|
| `PORTIA_Binary_Detection.zip`, `PORTIA_Single_Vessel_Classification.zip`, `PORTIA_Multi_Label_Monitoring.zip` | < 1 MB each | **Everything:** the train/val/test CSVs |
| `PORTIA_Metadata.zip` | tiny | dataset info (check for the hydrophone position) |
| `PORTIA_audio.zip` + `.z01`…`.z06` | 74.7 GB total (one split ZIP) | audio (sections 4–6) |

**Step 1: the CSVs (2.5 MB).** Unzip them into `data/portia/`, so you get
`data/portia/PORTIA_Binary_Detection/…csv` etc. Sections 1–3 (split check, passes, map) already work.

**Step 2: a slice of the audio.**
```bash
sudo apt install p7zip-full
# choose ~6 ship passes and a 1,500-window-per-split baseline subset -> data/portia/selected_windows.txt
python scripts/extract_windows.py --passes 6 --task binary --n-per-split 1500
```
Then either:
- **Cheap try:** download only `PORTIA_audio.zip` (10.3 GB, the last part, which holds the file index) and run
  `python scripts/extract_windows.py --passes 6 --task binary --n-per-split 1500 --archive /path/PORTIA_audio.zip`.
  7-Zip extracts the selected windows that sit in that part and reports the rest as missing. The notebook uses whatever is there.
- **Reliable:** download all 7 parts to an external drive (same folder), run the same command, then delete the parts.

Extracted windows go to `data/portia/audio/`.

## Run
```bash
python3 -m venv ~/portia-env && source ~/portia-env/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/portia_ais_walkthrough.ipynb
```
**No data yet?** Test everything on a small **fake** dataset with the same layout and columns:
```bash
python tests/make_fake_portia.py --out data/fake_portia
PORTIA_ROOT=data/fake_portia jupyter lab notebooks/portia_ais_walkthrough.ipynb
```
(The fake data is synthetic. Its 100% baseline score means nothing.)

## Notes and caveats
- **The hydrophone position in the notebook is a placeholder** near the Port of Valencia. Set `HYDRO_LAT/HYDRO_LON` from `PORTIA_Metadata` or the paper.
- PORTIA gives **distances, not ship positions**, so the map shows rings. Exact tracks need AIS positions from another source.
- Turning inter-hydrophone delays into **bearing** needs the array geometry from the paper.
- The Zenodo page lists both 32-bit and 16-bit PCM. `soundfile` reads either.
- Column names follow the Zenodo description (`window_id, start_utc, label_vessel, n_vessels, primary_class, primary_dist, primary_mmsi, <class>, dist_<class>_km`). If the released CSVs differ, adjust `portia_tools.load_manifests`.

## Layout
```
portia_tools.py                 loading CSVs, finding passes, joining windows, levels, features
scripts/extract_windows.py      choose windows (passes / benchmark subset) and extract them with 7-Zip
notebooks/portia_ais_walkthrough.ipynb   the walkthrough (built from notebooks/build_notebook.py)
tests/make_fake_portia.py       fake dataset for testing without the real data
data/                           not committed (git-ignored)
```
