# GOES Signatures for Flood Prediction

A research project that uses GOES-16 and GOES-19 satellite imagery (ABI-L2-MCMIPC) to identify spectral signatures associated with flood events across the contiguous United States (CONUS).

## Project Structure

```
flood-prediction/
├── data/
│   ├── raw/          # ground truth flood event records (parquet)
│   └── goes/         # downloaded GOES satellite imagery (NetCDF)
├── notebooks/
│   └── explore.ipynb # exploratory analysis of ground truth data
├── src/
│   └── goes_data.py  # GOES download script (CLI + importable module)
├── pyproject.toml
└── uv.lock
```

## Setup

Requires Python ≥ 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Data

### Ground Truth Flood Events

Download the flood event records and place them in `data/raw/`:

```bash
wget -O data/raw/groundsource_2026.parquet \
  "https://zenodo.org/records/18647054/files/groundsource_2026.parquet?download=1"
```

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
uv run python src/goes_data.py estimate

# 6 images/day
uv run python src/goes_data.py estimate --images-per-day 6
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
# Dry-run: see what would be downloaded without fetching anything
uv run python src/goes_data.py download --dry-run

# Full download (2020–2026-02-28, 1 image/day at 18:00 UTC)
uv run python src/goes_data.py download

# Custom date range or hour
uv run python src/goes_data.py download \
  --start-date 2023-01-01 --end-date 2023-12-31 \
  --workers 32

# 6 images/day (daytime hours)
uv run python src/goes_data.py download --hour 13 15 17 18 19 21
```

Downloads are resumable — already-downloaded files are skipped automatically. Files are saved to `data/goes/GOES{16|19}/YYYY/MM/DD/`.

## Notebooks

| Notebook | Description |
|---|---|
| [notebooks/explore.ipynb](notebooks/explore.ipynb) | Exploratory analysis of ground truth flood event data: spatial distribution, temporal coverage, area statistics |

## Band Reference

| Use case | Bands |
|---|---|
| GeoColor (true color) | 3, 2, 1 |
| Cloud properties | 6 |
| All bands | 1–16 (all in each ABI-L2-MCMIPC file) |
