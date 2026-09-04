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
| **Total footage** | 6 videos (~35 min) |

### Video files

| File | Duration | Vantage point | Description |
|---|---|---|---|
| `data/Record_1.mp4`  | 10m01s | 🚶 Ground level      | Recorded from street level at the intersection |
| `data/Record_2.mp4`  |  5m00s | 🚶 Ground level      | Recorded from street level at the intersection |
| `data/Record 3.mp4`  |  5m00s | 🏢 3rd-floor balcony | Overhead view from a building balcony |
| `data/Record 4.mp4`  |  5m00s | 🏢 3rd-floor balcony | Overhead view from a building balcony |
| `data/Record 5.mp4`  |  5m00s | 🏢 3rd-floor balcony | Overhead view from a building balcony |
| `data/Record 6.mp4`  |  5m00s | 🏢 3rd-floor balcony | Overhead view from a building balcony |

> **Note:** Records 1 & 2 were collected from the ground, providing a street-level perspective. Records 3–6 were collected from the balcony of a 3-storey building, offering a near-top-down view well-suited for trajectory extraction.
> The 30 FPS rate provides sub-33 ms temporal resolution — sufficient for fine-grained speed estimation.

### Map

**Coordinates:** 3°51'50.5"N  11°29'47.8"E (decimal: 3.864028, 11.496611)

[![Carrefour Melen - OpenStreetMap](https://staticmap.openstreetmap.de/staticmap.php?center=3.864028,11.496611&zoom=16&size=640x360&markers=3.864028,11.496611,ltblue)](https://www.openstreetmap.org/?mlat=3.864028&mlon=11.496611&zoom=16)

> Click the map to open interactively in OpenStreetMap. Also viewable on [Google Maps](https://maps.google.com/?q=3.864028,11.496611).


---

## Results

| Metric | Yaounde (balcony) | ETH/UCY |
|---|---|---|
| Mean speed (m/s) | _TBD_ | ~1.0 |
| Median linearity | _TBD_ | ~0.85 |
| Density (ped/m2) | _TBD_ | _TBD_ |

> Results will be updated once analysis is complete.

<!-- Uncomment when results are available:
![Tracking Demo](results/tracking_demo.gif)
![Heatmap](results/heatmap_balcony.png)
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
│
├── README.md
├── requirements.txt
│
├── src/                          # Source code
│   ├── run_all.py                # 🚀 Full pipeline orchestrator (all 6 recordings)
│   ├── track.py                  # YOLOv8 + ByteTrack — detection & tracking
│   ├── extract_traj.py           # Trajectory extraction to CSV + speed computation
│   ├── visualize.py              # Heatmaps, trajectory overlays, speed plots
│   ├── compare_eth_ucy.py        # Comparative analysis vs ETH/UCY benchmarks
│   └── make_gif.py               # Convert tracking video to animated GIF
│
├── data/                         # Raw videos & annotations
│   ├── Record_1.mp4              # 10m01s │ 🚶 Ground level
│   ├── Record_2.mp4              #  5m00s │ 🚶 Ground level
│   ├── Record 3.mp4              #  5m00s │ 🏢 3rd-floor balcony
│   ├── Record 4.mp4              #  5m00s │ 🏢 3rd-floor balcony
│   ├── Record 5.mp4              #  5m00s │ 🏢 3rd-floor balcony
│   ├── Record 6.mp4              #  5m00s │ 🏢 3rd-floor balcony
│   ├── trajectories/             # Per-recording & merged trajectory CSVs
│   │   ├── record_1.csv … record_6.csv
│   │   ├── all_ground.csv        # Records 1+2 merged
│   │   └── all_balcony.csv       # Records 3–6 merged  ← used for analysis
│   └── eth_ucy/                  # ETH/UCY benchmark annotations (.txt)
│       ├── biwi_eth.txt
│       ├── biwi_hotel.txt
│       ├── crowds_zara01.txt
│       ├── crowds_zara02.txt
│       ├── crowds_zara03.txt
│       ├── students001.txt
│       ├── students003.txt
│       └── uni_examples.txt
│
├── results/                      # Generated outputs
│   ├── tracking_demo.gif                 # Animated tracking preview
│   ├── heatmap_balcony.png               # Pedestrian density heatmap (balcony)
│   ├── trajectories_per_recording.png    # Trajectory grid (6 recordings)
│   ├── speed_ground_vs_balcony.png       # Ground vs balcony speed comparison
│   ├── speed_per_recording.png           # Per-recording speed vs ETH/UCY
│   ├── speed_comparison.png              # Yaounde (balcony) vs ETH/UCY
│   └── comparison_summary.csv           # Quantitative comparison table
│
└── notebooks/
    └── analysis.ipynb            # Interactive exploration & visualisation
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (recommended)

`run_all.py` handles tracking, speed extraction, and merging for all 6 recordings in one command:

```bash
# All recordings — ground + balcony
python src/run_all.py

# Balcony only (recommended for trajectory analysis & ETH/UCY comparison)
python src/run_all.py --groups balcony

# Low RAM? Reduce inference size
python src/run_all.py --groups balcony --imgsz 480
```

This produces:
- `data/trajectories/record_N.csv` — per-recording trajectories
- `data/trajectories/all_ground.csv` — Records 1+2 merged
- `data/trajectories/all_balcony.csv` — Records 3–6 merged

### 3. (Optional) Run a single recording manually

```bash
python src/track.py "data/Record 3.mp4" \
  --output-dir results/tracking/record_3 \
  --csv data/trajectories/record_3.csv

python src/extract_traj.py \
  --input data/trajectories/record_3.csv \
  --fps 30 --px-per-m 30
```

### 4. Compare with ETH/UCY

```bash
python src/compare_eth_ucy.py \
  --yaounde-csv data/trajectories/all_balcony.csv \
  --eth-dir data/eth_ucy \
  --yaounde-fps 30
```

### 5. Visualise

```bash
python src/visualize.py \
  --csv data/trajectories/all_balcony.csv \
  --frame-w 1920 --frame-h 1080
```

### 6. Generate demo GIF

```bash
python src/make_gif.py results/traking/Record_2.avi --output results/tracking_demo.gif
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

- Alahi, A.; Goel, K.; Ramanathan, V.; Robicquet, A.; Fei-Fei, L.; Savarese, S. *Social LSTM: Human Trajectory Prediction in Crowded Spaces*. **CVPR 2016**. [CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Alahi_Social_LSTM_Human_CVPR_2016_paper.pdf)
- Saadatnejad, S.; Gao, Y.; Messaoud, K.; Alahi, A. *Social-Transmotion: Promptable Human Trajectory Prediction*. **ICLR 2024** — VITA lab, EPFL. [arXiv](https://arxiv.org/abs/2312.16168) | [Code](https://github.com/vita-epfl/social-transmotion)
- Feudjio, S.L.T.; Tchaheu, D.T.; Fondzenyuy, S.K.; Jackai, I.N., II; Usami, D.S.; Persia, L. *Investigating and Improving Pedestrian Safety in an Urban Environment of a Low- or Middle-Income Country: A Case Study of Yaounde, Cameroon*. **Future Transportation, 2024**, 4, 548-578. DOI: [10.3390/futuretransp4020026](https://doi.org/10.3390/futuretransp4020026)

---

