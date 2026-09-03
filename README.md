# Pedestrian Tracking & Trajectory Analysis — Carrefour Melen, Yaounde, Cameroon

> **"Pedestrian Trajectory Extraction in Unstructured Urban Traffic: A Case Study from Yaounde, Cameroon"**

A computer-vision pipeline that **detects, tracks, and analyses pedestrian trajectories** from video footage filmed at **Carrefour Melen, Yaounde**, and compares the observed motion patterns against the ETH/UCY benchmark datasets used in state-of-the-art trajectory prediction research (Social Force, Social-Transmotion, etc.).

---

## Motivation

Pedestrian trajectory datasets from Low- and Middle-Income Countries (LMICs) are almost entirely absent from the literature. This project bridges that gap by:

1. Applying a **YOLOv8 + ByteTrack** pipeline to unstructured urban traffic footage from **Carrefour Melen, Yaounde**.
2. Extracting trajectories in the standard ETH/UCY CSV format (`id, frame, x, y`).
3. Quantitatively comparing speed distributions, path linearity, and pedestrian density with European datasets.

---

## Data Collection

| Parameter | Value |
|---|---|
| **Location** | Carrefour Melen, Yaounde, Cameroon |
| **Camera** | Samsung Galaxy A56 |
| **Resolution** | Full HD — 1920 × 1080 px |
| **Frame rate** | 30 FPS |
| **Files** | `data/Record_1.mp4` (10m01s, 1277 MB), `data/Record_2.mp4` (5m00s, 639 MB) |
| **Total footage** | ~15 minutes |

> Video was recorded handheld / from a fixed vantage point at the intersection.
> The 30 FPS rate provides sub-33 ms temporal resolution — sufficient for fine-grained speed estimation.

### Map

**Coordinates:** 3°51'50.5"N  11°29'47.8"E (decimal: 3.864028, 11.496611)

[![Carrefour Melen - OpenStreetMap](https://staticmap.openstreetmap.de/staticmap.php?center=3.864028,11.496611&zoom=16&size=640x360&markers=3.864028,11.496611,ltblue)](https://www.openstreetmap.org/?mlat=3.864028&mlon=11.496611&zoom=16)

> Click the map to open interactively in OpenStreetMap. Also viewable on [Google Maps](https://maps.google.com/?q=3.864028,11.496611).


---

## Results

| Metric | Yaounde | ETH/UCY |
|---|---|---|
| Mean speed (m/s) | _TBD_ | ~1.0 |
| Median linearity | _TBD_ | ~0.85 |
| Density (ped/m2) | _TBD_ | _TBD_ |

> Results will be updated once analysis is complete.

<!-- Uncomment when results are available:
![Tracking Demo](results/tracking_demo.gif)
![Heatmap](results/heatmap.png)
![Speed Comparison](results/speed_comparison.png)
-->

---

## Stack

| Component | Tool |
|---|---|
| Object detection | YOLOv8 (Ultralytics) |
| Multi-object tracking | ByteTrack (integrated in Ultralytics) |
| Trajectory extraction | Python + Pandas + NumPy |
| Visualisation | OpenCV + Matplotlib |
| Comparison dataset | ETH/UCY (open source) |

---

## Project Structure

```
pedestrian-tracking-yaounde/
+-- README.md
+-- requirements.txt
+-- src/
|   +-- track.py            # YOLOv8 + ByteTrack detection & tracking
|   +-- extract_traj.py     # Extract trajectories to CSV + compute speeds
|   +-- visualize.py        # Heatmaps, trajectory overlays, speed plots
|   +-- compare_eth_ucy.py  # Comparative analysis vs ETH/UCY
|   +-- make_gif.py         # Convert tracking video to demo GIF
+-- data/
|   +-- Record_1.mp4                # Collected at Carrefour Melen (Samsung A56, FHD 30fps)
|   +-- Record_2.mp4
|   +-- trajectories_yaounde.csv    # Generated output
|   +-- eth_ucy/                    # Place ETH/UCY .txt files here
+-- results/
|   +-- tracking_demo.gif
|   +-- heatmap.png
|   +-- trajectories.png
|   +-- speed_distribution.png
|   +-- speed_comparison.png
+-- notebooks/
    +-- analysis.ipynb      # Interactive exploration
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run tracking on your video

```bash
# Record_1
python src/track.py data/Record_1.mp4 --output-dir results

# Record_2
python src/track.py data/Record_2.mp4 --output-dir results
```

### 3. Extract trajectories

```bash
python src/extract_traj.py --raw-results results/raw_results.pkl --fps 30 --output data/trajectories_yaounde.csv
```

### 4. Visualise

```bash
python src/visualize.py --csv data/trajectories_yaounde.csv --frame-w 1920 --frame-h 1080
```

### 5. Compare with ETH/UCY

Download ETH/UCY annotations from [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus/tree/master/experiments/pedestrians/raw/raw/all_data) and place `.txt` files in `data/eth_ucy/`, then:

```bash
python src/compare_eth_ucy.py --yaounde-csv data/trajectories_yaounde.csv --eth-dir data/eth_ucy --yaounde-fps 30
```

### 6. Generate demo GIF

```bash
python src/make_gif.py results/tracking_run/Record_1.mp4 --output results/tracking_demo.gif
```

### 7. Interactive notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## ETH/UCY Data

Download annotations from:
- [Trajectron++ repository](https://github.com/StanfordASL/Trajectron-plus-plus/tree/master/experiments/pedestrians/raw/raw/all_data)
- Or the [original ETH dataset](https://icu.ee.ethz.ch/research/datsets.html)

Place `.txt` files in `data/eth_ucy/`.

Format: `frame_id  person_id  x  y` (space-separated, positions in metres).

---

## Observations

> *To be filled in after data collection and analysis.*

Key hypotheses to test:
- Yaounde pedestrians move **faster** and more **erratically** than ETH/UCY counterparts.
- Trajectory **linearity** is lower due to obstacle avoidance (motos, vendors, uneven surfaces).
- **Local density** is higher at intersections but interaction models (Social Force, etc.) may not generalise.
- The presence of **moto-taxis** (benskins) creates pedestrian-vehicle interactions absent from ETH/UCY.

---

## Related Work

- Alahi et al., *Social Force* (CVPR 2014)
- Xu et al., *Social-Transmotion* (ICLR 2024) — VITA lab, EPFL
- Feudjio, S.L.T.; Tchaheu, D.T.; Fondzenyuy, S.K.; Jackai, I.N., II; Usami, D.S.; Persia, L. *Investigating and Improving Pedestrian Safety in an Urban Environment of a Low- or Middle-Income Country: A Case Study of Yaounde, Cameroon*. **Future Transportation, 2024**, 4, 548-578. DOI: [10.3390/futuretransp4020026](https://doi.org/10.3390/futuretransp4020026)

---

*Video data collected at Carrefour Melen, Yaounde, Cameroon — Samsung Galaxy A56, 1920x1080 @ 30 FPS.*
*Part of a research demonstration for the VITA lab (Prof. Alahi), EPFL.*
