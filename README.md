# GOES Signatures for Flood Prediction

A research project that uses GOES-16 and GOES-19 satellite imagery (ABI-L2-MCMIPC) to identify spectral signatures associated with flood events across the contiguous United States (CONUS).

> **Status: early research — the prediction target is not finalized.** The current
> focus is exploratory: watching a day's GOES cloud time-lapse over CONUS against the
> *next day's* flood polygons on one interactive map, to see whether cloud and
> moisture movement lines up with where extreme flooding appears. Two flood
> ground-truth layers back this up — observed flood extents (**groundsource**) and
> NWS flood **warnings** (flash-flood + areal-flood) — merged into a single unified
> parquet.

## Project Structure

```
flood-prediction/
├── data/
│   ├── raw/                # groundsource flood-extent parquet (downloaded, gitignored)
│   └── flood_warnings/     # floods_unified.parquet (committed) + build intermediates
│                           # GOES imagery (NetCDF) downloads to /mnt/disk1/goes-data/
├── notebooks/
│   ├── explore_flood_data.ipynb  # load/summarize groundsource + warnings; unified frame
│   ├── clouds_vs_floods.ipynb    # GOES time-lapse + 25km grid + next-day flood overlay
│   └── preview_goes.ipynb        # interactive viewer for downloaded GOES imagery
├── src/
│   ├── download_goes.py            # GOES download script (CLI + importable module)
│   └── download_flood_data.py      # NWS FF/FA warnings (IEM) + groundsource fetch (Zenodo)
├── pyproject.toml
└── uv.lock
```

## Setup

Requires Python ≥ 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
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

Two complementary flood layers, both fetched via `src/download_flood_data.py`:

```bash
# groundsource flood-extent polygons (~637 MB) from Zenodo -> data/raw/
uv run python src/download_flood_data.py groundsource

# NWS Flash Flood + Areal Flood warning polygons (CONUS, 2019-2026) from IEM
# -> data/flood_warnings/flood_warnings_conus.parquet
uv run python src/download_flood_data.py warnings
```

Both are idempotent/resumable: `warnings` caches each fetched polygon so re-runs
only fetch what's missing, and `groundsource` skips if the file is already present
(`--force` to re-download). `explore_flood_data.ipynb` merges the two into the
**unified** layer at `data/flood_warnings/floods_unified.parquet` — the one flood
artifact committed to the repo, so you can use it directly without re-downloading or
rebuilding.

### GOES Satellite Imagery

Imagery is downloaded from NOAA's public AWS S3 buckets (`noaa-goes16`, `noaa-goes19`). No AWS credentials are required.

**Satellite coverage:**
| Period | Satellite | S3 bucket |
|---|---|---|
| 2020-01-01 → 2025-04-06 | GOES-16 | `noaa-goes16` |
| 2025-04-07 → present | GOES-19 | `noaa-goes19` |

**Product:** `ABI-L2-MCMIPC` — multi-band cloud & moisture imagery, CONUS sector, all 16 ABI bands in a single NetCDF file (~60 MB/file).

#### Estimate storage before downloading

```bash
# 1 image/day (default)
uv run python src/download_goes.py estimate

# 6 images/day
uv run python src/download_goes.py estimate --images-per-day 6
```

Expected output (2020 – 2026-02-28):

```
                     GOES-16    GOES-19      Total
Days with data         1,923        328      2,251
Files @ 1/day          1,923        328      2,251
Storage @ 1/day       112.7G      19.2G     131.9G

--- 6 images/day estimate ---
Files                 11,538      1,968     13,506
Storage               676.1G     115.3G     791.4G
```

#### Download

```bash
# Dry-run: list every file and report the EXACT total size (no fetching)
uv run python src/download_goes.py download --dry-run

# Full download — default: 6 daytime images/day (16-21 UTC) -> /mnt/disk1/goes-data
uv run python src/download_goes.py download

# Custom date range or hour
uv run python src/download_goes.py download \
  --start-date 2023-01-01 --end-date 2023-12-31 \
  --workers 32

# 1 image/day, or a custom location (override the 6/day + path defaults)
uv run python src/download_goes.py download --hour 18 --data-dir /some/other/path
```

Downloads are resumable — already-downloaded files are skipped automatically. Files are saved to `/mnt/disk1/goes-data/GOES{16|19}/YYYY/MM/DD/` by default (override with `--data-dir`).

> **Tip:** `--dry-run` reports the **exact** total download size, summed from real S3 object sizes (nothing is fetched). Use it when you need an accurate figure; `estimate` is a faster rough projection at ~60 MB/file.

## Workflow

1. **Flood ground truth** — `download_flood_data.py groundsource` + `warnings`, or
   just use the committed `data/flood_warnings/floods_unified.parquet`.
2. **GOES imagery** — `download_goes.py download` for the dates of interest.
3. **Explore** — `explore_flood_data.ipynb` summarizes both layers and (re)builds the
   unified parquet.
4. **Compare** — `clouds_vs_floods.ipynb` overlays the GOES time-lapse, the 25 km
   CONUS grid, and the next-day floods on one interactive map.

## Notebooks

Launch with `uv run jupyter lab`.

| Notebook | Description |
|---|---|
| [notebooks/explore_flood_data.ipynb](notebooks/explore_flood_data.ipynb) | Load and summarize both flood layers (groundsource extents + NWS FF/FA warnings), then build one harmonized GeoDataFrame (`data/flood_warnings/floods_unified.parquet`) keyed by issue/expire date, area, and geometry |
| [notebooks/clouds_vs_floods.ipynb](notebooks/clouds_vs_floods.ipynb) | Watch a day's GOES time-lapse against the next day's floods: reprojected cloud frames + a 25 km CONUS land grid + the unified flood layer (groundsource / flash-flood / areal-flood) as toggleable overlays on one scroll-zoom map |
| [notebooks/preview_goes.ipynb](notebooks/preview_goes.ipynb) | Preview downloaded GOES imagery: pick a date/file, view any band or a true-color RGB, then overlay it on an interactive folium map (scroll-zoom/pan, switchable basemaps) reprojected to lat/lon; crop to a lon/lat or pixel box; plus quick static previews |

## Band Reference

| Use case | Bands |
|---|---|
| True color (RGB = R, G, B) | 2, 3, 1 |
| Cloud properties | 6 |
| All bands | 1–16 (all in each ABI-L2-MCMIPC file) |
