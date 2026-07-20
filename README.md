# GOES Signatures for Flood Prediction

A research project that uses **GOES satellite imagery** and **GLM lightning data**
to predict **which areas of the contiguous US (CONUS) will flood the next day**, on
a **50 × 50 km grid**.

**At a glance:**

- **Training range:** **2019–2025** (all 7 years; the model is trained and evaluated
  with blocked K-fold cross-validation across the full span). 2026 is held out of
  training — its labels are still truncated.
- **Inputs:** one full day of **GOES ABI** imagery (8 frames/day, 5 emissive-IR
  bands) **+ GLM lightning**, downscaled from the raw ~1500 × 2500 pixel frame to a
  **50 × 50 km CONUS-land grid** of per-cell features (see *Downscaling* below).
- **Output:** a per-cell flood probability for the **next day** — which 50 km cells
  face flooding on CDT day **D** given day **D−1**'s observations.
- **Labels (ground truth):** *observed* floods = news-report extents
  (**groundsource**) ∪ NCEI **storm events**. NWS flood **warnings** are held out as
  a separate operational baseline to benchmark against, not used as a label.
- **Result:** the model finds real flood signal a day ahead and is **largely
  complementary** to same-day NWS warnings — together they cover more observed floods
  than either alone (see *Model vs. NWS warnings* below).

## Project Structure

```
goes-signatures-for-flood/
├── data/
│   ├── raw/                # groundsource flood-extent parquet (downloaded, gitignored)
│   ├── flood_warnings/     # floods_unified.parquet (committed) + build intermediates
│   └── storm_events/       # storm_events_flood.parquet (NCEI observed flood reports)
│                           # GOES imagery (NetCDF) lives on /mnt/disk4/goes-data/
├── floodlens/                      # importable package (editable-installed via pyproject)
│   ├── config.py                   # shared constants (paths, bands, grid, cache/outputs)
│   ├── gridindex.py                # GOES-pixel -> 50 km-cell index (shared)
│   ├── foldsplit.py                # blocked K-fold CV splits (fixed test + rotating val)
│   ├── run_cv.py                   # CV sweep driver: train each model on every fold (DDP)
│   ├── trainers/                   # resnet3d, cnn_attn, convgru_attn, xgb
│   └── download/                   # goes, flood_data, glm (data downloaders, CLI)
├── notebooks/
│   ├── explore/
│   │   ├── goes_data_explore.ipynb   # GOES imagery: disk inventory, bands, single-frame map
│   │   ├── flood_data_explore.ipynb  # groundsource + warnings; builds floods_unified.parquet
│   │   ├── glm_data_explore.ipynb    # GLM flashes: availability, daily counts, density
│   │   └── goes_vs_floods.ipynb      # combined overlay + time-lapse (clouds, floods, lightning)
│   └── model/                      # the .ipynb pipeline only (imports from floodlens)
│       ├── 01_prepare_data.ipynb     # build the 50 km feature-grid sample cache (inputs + labels)
│       ├── 02_model_comparison.ipynb # compare models under K-fold CV (metrics + curves + maps)
│       └── 03_vs_nws_warnings.ipynb  # benchmark the model vs NWS flood warnings
├── outputs/                        # model checkpoints + results (gitignored)
├── StormArthurEvaluation/          # standalone side experiment (June 2026 same-day model)
├── pyproject.toml
└── uv.lock
```

## Setup

Requires Python ≥ 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync   # also installs the `floodlens` package editable, so notebooks/scripts can `import floodlens`
```

Notebook outputs are **not** tracked in git (a [nbstripout](https://github.com/kynan/nbstripout)
filter strips them on commit while leaving them in your working copy). After cloning,
enable it once:

```bash
.venv/bin/python -m pip install nbstripout
.venv/bin/nbstripout --install --attributes .gitattributes
```

### GPU / ML stack

This project targets a CUDA GPU workstation (developed on 2× NVIDIA RTX PRO 6000
Blackwell). `uv sync` installs the GPU/ML stack — **PyTorch (CUDA 12.8 wheels),
Lightning, CuPy**, plus `timm`, `einops`, `xbatcher`, and `tensorboard`. The cu128
wheels are required for Blackwell (sm_120) GPUs; the PyTorch index is pinned in
`pyproject.toml`. Both GPUs are usable for data-parallel (DDP) training.

RAPIDS (GPU dataframes + spatial + ML) is installed separately — it can't be locked
cleanly and pins numpy down:

```bash
.venv/bin/python -m pip install --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.4.*" "cuspatial-cu12==25.4.*" "cuml-cu12==25.4.*"
```

> A plain `uv sync` drops this overlay and bumps numpy back; re-run the line above
> afterward, or use `uv sync --inexact`. See [CLAUDE.md](CLAUDE.md) for full details.

## Data

### Flood ground truth

Three complementary flood layers, all fetched via `floodlens.download.flood_data`
(`all` runs the three in sequence):

```bash
# groundsource flood-extent polygons (~637 MB) from Zenodo -> data/raw/
uv run python -m floodlens.download.flood_data groundsource

# NWS Flash Flood + Areal Flood warning polygons (CONUS, 2019-2026) from IEM
# -> data/flood_warnings/flood_warnings_conus.parquet
uv run python -m floodlens.download.flood_data warnings

# NCEI Storm Events flood reports (observed occurrences, points)
# -> data/storm_events/storm_events_flood.parquet
uv run python -m floodlens.download.flood_data storms

# everything above
uv run python -m floodlens.download.flood_data all
```

The three layers differ in nature — keep that in mind when using them as labels:

| layer | what it is | geometry | time resolution |
|---|---|---|---|
| groundsource | flood events from news reports ([Google Research](https://zenodo.org/records/18647054)) | polygons | day (start/end date) |
| warnings | forecaster-issued *predictions* | polygons | minute (issue/expire) |
| storm events | human-confirmed *occurrences* with impacts | points | minute (UTC begin/end) |

**Storm events** come from the NCEI Storm Events Database bulk CSVs (details +
locations tables, re-stamped monthly by NCEI): flood-related event types (Flash
Flood / Flood / Heavy Rain / Debris Flow) with UTC times, flood cause, damage
estimates, injuries/deaths, report source (emergency managers, gauges, law
enforcement, public, ...), and narratives — one row per (event, location point),
CONUS, 2019 on (~55k events / ~183k points, ~11 MB). They are *observations*,
not warnings — though not fully independent of the warning process, since NWS
offices compile them partly to verify their own warnings (in 2019–2026, ~72% of
flood/flash-flood events fall inside an active warning, while only ~45% of
warnings contain an observed event).

All subcommands are idempotent/resumable: `warnings` caches each fetched polygon,
`storms` caches the raw NCEI files by version stamp, and `groundsource` skips if
present (`--force` to re-download). `flood_data_explore.ipynb` merges groundsource
+ warnings into the **unified** layer at `data/flood_warnings/floods_unified.parquet`
— the one flood artifact committed to the repo, so you can use it directly without
re-downloading or rebuilding.

### GOES Satellite Imagery

Imagery is downloaded from NOAA's public AWS S3 buckets (`noaa-goes16`, `noaa-goes19`). No AWS credentials are required.

**Satellite coverage:**
| Period | Satellite | S3 bucket |
|---|---|---|
| 2019-01-01 → 2025-04-06 | GOES-16 | `noaa-goes16` |
| 2025-04-07 → present | GOES-19 | `noaa-goes19` |

**Product:** `ABI-L2-MCMIPC` — multi-band cloud & moisture imagery, CONUS sector, all 16 ABI bands in a single NetCDF file (~60 MB/file).

#### Estimate storage before downloading

```bash
# 1 image/day (default)
uv run python -m floodlens.download.goes estimate

# the model pipeline's cadence: 8 frames/day over the training span
uv run python -m floodlens.download.goes estimate \
  --start-date 2019-01-01 --end-date 2025-12-31 --images-per-day 8
```

Expected output for the 8-frame/day pull over the 2019–2025 training span:

```
GOES Storage Estimate
Date range  : 2019-01-01 to 2025-12-31 (2557 days)
File size   : ~60 MB per ABI-L2-MCMIPC file
Images/day  : 8

                        GOES-16    GOES-19      Total
Days with data            2,288        269      2,557
Files                    18,304      2,152     20,456
Storage                 1072.5G     126.1G    1198.6G
```

#### Download

```bash
# Dry-run: list every file and report the EXACT total size (no fetching)
uv run python -m floodlens.download.goes download --dry-run

# Full download — default: all hourly frames -> /mnt/disk4/recent-goes
uv run python -m floodlens.download.goes download

# Custom date range or hour
uv run python -m floodlens.download.goes download \
  --start-date 2023-01-01 --end-date 2023-12-31 \
  --workers 32

# 1 image/day, or a custom location (override the frames + path defaults)
uv run python -m floodlens.download.goes download --hour 18 --data-dir /some/other/path
```

Downloads are resumable — already-downloaded files are skipped automatically. Files are saved to `/mnt/disk4/recent-goes/GOES{16|19}/YYYY/MM/DD/` by default (override with `--data-dir`). The model pipeline reads its 8-frame/day GOES from `/mnt/disk4/goes-data` (`config.DATA_DIR`).

> **Tip:** `--dry-run` reports the **exact** total download size, summed from real S3 object sizes (nothing is fetched). Use it when you need an accurate figure; `estimate` is a faster rough projection at ~60 MB/file.

## Workflow

1. **Flood ground truth** — `download_flood_data.py all` (groundsource + warnings +
   storm events), or just use the committed `data/flood_warnings/floods_unified.parquet`.
2. **GOES imagery** — `download_goes.py download` for the dates of interest.
3. **Explore** — the `notebooks/explore/` notebooks summarize each dataset;
   `flood_data_explore.ipynb` (re)builds the unified parquet and cross-checks the
   flood layers against each other.
4. **Compare** — `explore/goes_vs_floods.ipynb` overlays the GOES time-lapse, the
   CONUS grid, and the next-day floods on one interactive map.
5. **Model** — the `notebooks/model/` pipeline: `01_prepare_data` builds the
   feature-grid cache, `floodlens.run_cv` trains the model suite across all CV folds, and
   `02_model_comparison` / `03_vs_nws_warnings` evaluate them (see **Modeling** below).

## Modeling

**Task (v1).** From a CDT day **D−1** GOES/GLM observation, predict which **50 km
CONUS-land cells** are **flooded** on day **D**.

**Training range.** **2019–2025** — the full 7 years, used together via blocked
K-fold cross-validation (below). **2026 is excluded from training** (its
groundsource labels stop 2026-01-28 and NCEI storm events lag by months, so its base
rate is a truncated-label artifact); it is kept only as an extra held-out set for the
warnings benchmark.

**Inputs (per sample).** One CDT day **D−1** of observations:

- **GOES ABI** — **8 frames/day** (full UTC day, 3-hourly at 00,03,…,21 UTC) so both
  day and night are covered, using **5 all-emissive-IR bands** (8, 10, 11, 14, 15 —
  water vapour + cloud-top thermodynamics, readable around the clock).
- **GLM lightning** — per-day flash counts / density / occurrence.
- Plus per-cell **daily summaries** and a **lead-time** channel.

**Output (labels).** The target is *observed* floods on day **D** — groundsource
news-report extents **∪** NCEI storm events — rasterized to the grid as a per-cell
0/1 map. The model emits a **per-cell flood probability**; NWS warnings are **not**
used as a label (they are the benchmark, below).

**Downscaling to the 50 × 50 km grid.** Each raw ABI frame is ~**1500 × 2500 pixels**
(2 km). We project every pixel to CONUS Albers and pool it into a **59 × 95** grid of
**50 km** equal-area cells (3,360 CONUS-land cells; `gridindex.build_pix2cell` +
`config.build_grid_cells`). `01_prepare_data.ipynb` materializes the per-cell
feature-grid cache (`cache/goes_grid50_2019_2026/`): **8 frames/day × 19 per-cell
GOES/GLM features** + daily summaries + lead time. So the model operates on compact
per-cell *signatures*, not full-resolution imagery.

**Evaluation — blocked K-fold cross-validation** (`foldsplit.py`). One **fixed
held-out test set** (~10% of months, interleaved so it spans all of 2019–2025) plus
**6 CV folds** over the rest: each fold rotates one block as validation (~15%) and
trains on the others (~75%), with a ±3-day buffer so no multi-day event straddles a
train/eval boundary. Every split spans all 7 years with matched base rates, so results
are reported as **mean ± std across folds** plus a **6-fold ensemble**.

**Models (4).** Three deep encoders over the feature grid — a small 3D-ResNet
(`resnet3d`), a per-frame CNN + temporal attention (`cnn_attn`), and a ConvGRU +
attention (`convgru_attn`) — plus an **XGBoost** per-cell baseline (`xgb`). Each trainer
is standalone and reads the shared cache; deep models normalize inputs with per-fold,
train-only statistics.

**Run the sweep** (4 models × 6 folds = 24 runs; deep models train across both GPUs via
DDP, tabular on CPU):

```bash
python -m floodlens.run_cv                  # full sweep (resumable; skips finished folds)
python -m floodlens.run_cv --models xgb     # just the tabular baseline, all folds
python -m floodlens.run_cv --force          # retrain everything
```

Artifacts land in `outputs/` (`<name>_f<k>.pt`/`.pkl` + deep
`<name>_f<k>_results.npz`, all gitignored). Then open `02_model_comparison.ipynb` for
the CV metrics table (AUPRC mean ± std + ensemble) and `03_vs_nws_warnings.ipynb` for
the operational-warning benchmark.

### Model vs. NWS warnings

`03_vs_nws_warnings.ipynb` benchmarks the committed model (the **6-fold ConvGRU+attn
ensemble**) against the operational forecaster baseline: **NWS Flash-Flood + Areal-Flood
warnings**. Both are rasterized to the same 50 km grid on the fixed test set and scored
against the observed-flood labels. Note the lead-time gap — the model predicts day **D**
from **D−1** imagery (~1-day lead), while warnings are same-day nowcasts — so this is
*complementary*, not apples-to-apples.

On the fixed test set (271 days, base rate 1.67%):

| predictor | precision | recall | F1 | CSI | AUPRC |
|---|---|---|---|---|---|
| ConvGRU+attn (D−1 GOES) | 0.19 | 0.19 | 0.19 | 0.11 | **0.12 (7.2× base)** |
| NWS warnings (same-day) | 0.46 | 0.28 | 0.35 | 0.21 | — (binary) |

**The model complements NWS warnings.** They catch *different* floods, not the same
ones — of all observed flooded cell-days on the test set:

- NWS warnings alone: **28.4%**
- ConvGRU+attn alone: **19.3%**
- **Model *or* NWS: 39.1%** — the model adds **~11 points** of coverage beyond warnings
- Model *and* NWS (overlap): only **8.7%**

Day-to-day it cuts both ways: on some days the model beats NWS (e.g. 2023-03-13, CSI
0.33 vs 0.00) and on others NWS beats the model (e.g. 2024-01-09, 0.17 vs 0.47). A
day-ahead GOES/GLM signature and a same-day warning are picking up largely
non-overlapping flood events, so the union covers more than either baseline alone —
the practical case for the model as an early, complementary screen.

## Notebooks

Launch with `uv run jupyter lab`.

| Notebook | Description |
|---|---|
| [notebooks/explore/flood_data_explore.ipynb](notebooks/explore/flood_data_explore.ipynb) | Load and summarize the flood layers (groundsource extents + NWS FF/FA warnings), build the harmonized `floods_unified.parquet`, and verify warnings against observations (per-warning verification + a 50 km cell-day classification report) |
| [notebooks/explore/goes_data_explore.ipynb](notebooks/explore/goes_data_explore.ipynb) | Explore downloaded GOES imagery: disk inventory, pick a date/file, view any band or a true-color RGB, overlay a frame on an interactive folium map, crop to a region |
| [notebooks/explore/glm_data_explore.ipynb](notebooks/explore/glm_data_explore.ipynb) | Explore GLM lightning flashes: days built so far, flashes/day time series, one day in detail (stats, diurnal cycle, spatial density) |
| [notebooks/explore/goes_vs_floods.ipynb](notebooks/explore/goes_vs_floods.ipynb) | Watch a day's GOES time-lapse against the next day's floods: reprojected cloud frames + a CONUS land grid + the unified flood layer + synced GLM lightning dots as toggleable overlays on one scroll-zoom map |
| [notebooks/model/01_prepare_data.ipynb](notebooks/model/01_prepare_data.ipynb) | Build the model's inputs (CDT day D−1 GOES/GLM per-cell features, 8 frames/day + daily summaries + lead time) and labels (day-D observed floods on the 50 km grid), printing every shape, then materialize the fast on-disk feature-grid cache |
| [notebooks/model/02_model_comparison.ipynb](notebooks/model/02_model_comparison.ipynb) | Compare the model suite under 6-fold CV: reload every fold checkpoint, report AUPRC mean ± std (val for selection, fixed test for the honest comparison) + a 6-fold ensemble, plus per-fold training curves and prediction maps |
| [notebooks/model/03_vs_nws_warnings.ipynb](notebooks/model/03_vs_nws_warnings.ipynb) | Benchmark the model (K-fold ensemble) against NWS Flash-Flood + Areal-Flood warnings on the fixed test set: metrics vs observed floods, who-catches-what coverage, per-day maps, and aggregate spatial views |

## Band Reference

| Use case | Bands |
|---|---|
| True color (RGB = R, G, B) | 2, 3, 1 |
| Cloud properties | 6 |
| All bands | 1–16 (all in each ABI-L2-MCMIPC file) |
