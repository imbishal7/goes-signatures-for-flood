"""Project-wide configuration shared across notebooks and scripts.

Import the constants you need, e.g.::

    from config import CACHE_DIR, BANDS, CELL_KM

The single most important lever is ``CELL_KM`` — the output cell size (km).
Everything downstream (cell polygons, label maps, the pixel->cell pooling index)
is derived from it, so changing it here changes the whole project consistently.
"""

from pathlib import Path

# repo root (this file lives at the root)
ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Storage — large data/artifacts live on /mnt/disk1 (see CLAUDE.md)
# ---------------------------------------------------------------------------
DATA_DIR = Path("/mnt/disk4/goes-data")          # GOES ABI NetCDFs (full-day 3-hourly)
GLM_DIR = Path("/mnt/disk1/glm-data")            # GLM lightning parquet/day
STATES_GEOJSON = DATA_DIR / "aux/us-states.geojson"       # CONUS land boundary
UNIFIED_PARQUET = ROOT / "data/flood_warnings/floods_unified.parquet"

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
# Full dataset — 2019-2026, all covered by full-day 3-hourly GOES on /mnt/disk4.
# Canonical split: train 2019-2024, validate 2025, test 2026 (assigned in nb 01).
YEARS = (2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)
YEAR = YEARS[0]            # back-compat alias for single-year code paths (explore nbs)

# GOES inputs: 5 ABI bands, ALL emissive infrared (brightness temperature, K) so they
# are observable day AND night. This matters now that one input day spans the FULL UTC
# day (00–21 UTC, 8 frames) including night: the reflectance bands (1/2/3/6/7-day) go
# dark after sunset and would be blank for ~half the frames, so we use only emissive
# channels that read a real signal around the clock.
#    8  upper-trop water-vapour 6.2um -> mid/upper moisture + jet-level dynamics
#   10  low-trop water-vapour  7.3um  -> low/mid-trop moisture = the fuel for heavy rain
#   11  cloud-top phase        8.4um  -> ice vs liquid cloud tops
#   14  IR longwave window    11.2um  -> cloud-top temperature = convective intensity
#   15  "dirty" longwave win  12.3um  -> low-level moisture via the 14-15 split-window
# Coverage: moisture (8, 10, 15), cloud-top thermodynamics (11, 14), convective depth
# (14). Visible/near-IR surface contrast is sacrificed for night coverage; with a
# 24-hour cadence the round-the-clock IR bands are the natural choice.
BANDS = (8, 10, 11, 14, 15)
N_BAND = len(BANDS)

# per-frame GOES channels: the 5 bands, optionally + a per-frame lead-time channel
#   USE_TIME — append each frame's lead time (whole hours before day D's UTC start,
#              i.e. D 00:00 UTC, from _t.npy) as one extra channel, so the model knows
#              how far ahead of the flood each frame sits.
USE_TIME = True
N_CH = N_BAND + (1 if USE_TIME else 0)            # GOES channels per frame (= 6)
T_FRAMES = 8                                      # 3-hourly frames/day, full UTC day (00,03,..,21Z)
IMG_H, IMG_W = 1500, 2500                         # ABI CONUS 2 km grid (rows, cols)

# GLM lightning (DEFERRED): a SEPARATE input stream (not a GOES channel) — the
# full input day binned into GLM_HOURS hourly maps on the CELL_KM grid, GLM_FEATS
# features each [flash count, total energy, total area], plus per-hour lead times.
# Not built in nb1 and excluded from the model for now (the `area` feature is
# mis-scaled at the source for ~Jan 2019); constants kept for when we re-add it.
USE_GLM = False
GLM_HOURS = 24
GLM_FEATS = 3                                      # count, energy, area

# >>> OUTPUT GRID SIZE LEVER <<<
# Square output cells CELL_KM on a side, generated directly over CONUS land in
# an equal-area projection (see build_grid_cells). Any size works: 50, 40, 75...
CELL_KM = 50

# ---- model (notebook 02) ----
# A shared per-frame conv encoder downsamples to IMG // POOL_STRIDE, a ConvLSTM
# fuses the T frames, then pixel->cell pooling maps onto the CELL_KM grid.
POOL_STRIDE = 2

# ---------------------------------------------------------------------------
# Derived artifact paths (depend on YEARS / CELL_KM so changing the grid or the year
# span never silently reuses a stale cache)
# ---------------------------------------------------------------------------
_YEAR_TAG = "_".join(str(y) for y in YEARS)       # e.g. "2019_2020"
STATS_PATH = DATA_DIR / f"aux/band_stats_{_YEAR_TAG}.json"
# Single canonical cache (built by nb 01): pure per-cell 50 km GOES/GLM signature features
# (seq + daily summaries) for 2019-2026 - no full-resolution image. ~9 GB, root NVMe.
CACHE_DIR = ROOT / "cache" / "goes_grid50_2019_2026"   # project NVMe (fast reads)
POOL_IDX_PATH = CACHE_DIR / f"pool_index_s{POOL_STRIDE}_{CELL_KM}km.npy"
CKPT_DIR = Path("/mnt/disk1/models/floodnet_convlstm")

# ---------------------------------------------------------------------------
# Train / val / test split (random, reproducible) — notebook 01
# ---------------------------------------------------------------------------
SPLIT_SEED = 0
SPLIT_FRACS = (0.70, 0.20, 0.10)                  # train / val / test

# ---------------------------------------------------------------------------
# Training hyper-parameters — notebook 02
# ---------------------------------------------------------------------------
BATCH_PER_GPU = 1                             # at POOL_STRIDE=2 (finer /2 encoder grid, ~4x activation memory)
EPOCHS = 2
LR = 3e-4
WORKERS = 8

# ---------------------------------------------------------------------------
# Loss & metrics — Tversky (soft IoU/F1) on hard 0/1 targets
# ---------------------------------------------------------------------------
# loss = 1 - TP / (TP + ALPHA*FP + BETA*FN), over the land cells.
#   ALPHA weights false positives, BETA weights false negatives:
#     ALPHA > BETA -> penalize over-prediction (precision-favoring)
#     ALPHA = BETA -> Dice loss
#     ALPHA < BETA -> favor recall (catch more, risk over-prediction)  [current]
# The target is extremely rare-positive (storm-event floods ~0.2% of land cells/day),
# so a precision-favoring loss (ALPHA>BETA) collapses to predicting all-zero. We
# favor RECALL (ALPHA<BETA) so the model actually fires on flood cells.
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
PRED_THRESHOLD = 0.5     # decision threshold for the binary val metrics (F1/IoU/...)


# ---------------------------------------------------------------------------
# Output grid — generated fresh on every call, never stored. Square CELL_KM
# cells laid over CONUS land in equal-area; a pure function of CELL_KM + the
# CONUS boundary, so both notebooks rebuild the identical grid from here.
# ---------------------------------------------------------------------------
_STATES_URL = ("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
               "master/data/geojson/us-states.json")
_NON_CONUS = {"Alaska", "Hawaii", "Puerto Rico"}   # dropped from the land grid
_ALBERS = 5070                                     # CONUS Albers equal-area (m)


def build_grid_cells():
    """Generate the CELL_KM square-cell grid over CONUS land (north-up).

    CELL_KM-metre cells are laid in CONUS Albers (EPSG:5070, so they are
    genuinely CELL_KM on a side) and kept where they intersect US land (lower-48
    + DC, from STATES_GEOJSON — downloaded once if missing). Nothing is stored.

    Returns
        cells      GeoDataFrame[R, C, cell_id, geometry] in EPSG:4326 (row 0 =
                   north, col 0 = west — matching the GOES imagery orientation).
        grid_r,    grid_c : int grid dimensions.
        land_mask  (grid_r, grid_c) bool — cells intersecting land.
    """
    import urllib.request

    import geopandas as gpd
    import numpy as np
    from shapely.geometry import box

    if not STATES_GEOJSON.exists():
        STATES_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_STATES_URL, STATES_GEOJSON)
    states = gpd.read_file(STATES_GEOJSON)
    land = states[~states["name"].isin(_NON_CONUS)].to_crs(_ALBERS)
    land_union = land.union_all()

    step = CELL_KM * 1000.0
    minx, miny, maxx, maxy = land.total_bounds
    x0s = np.arange(np.floor(minx / step) * step, maxx + step, step)   # W -> E
    y0s = np.arange(np.floor(miny / step) * step, maxy + step, step)   # S -> N
    grid_r, grid_c = len(y0s), len(x0s)

    recs = []
    for k, y0 in enumerate(y0s):
        R = grid_r - 1 - k                           # flip so row 0 = north
        for c, x0 in enumerate(x0s):
            recs.append({"R": R, "C": c,
                         "geometry": box(x0, y0, x0 + step, y0 + step)})
    grid = gpd.GeoDataFrame(recs, crs=_ALBERS)
    grid = grid[grid.intersects(land_union)].reset_index(drop=True)
    grid["cell_id"] = ("r" + grid["R"].map("{:03d}".format)
                       + "c" + grid["C"].map("{:03d}".format))

    land_mask = np.zeros((grid_r, grid_c), dtype=bool)
    land_mask[grid["R"], grid["C"]] = True
    return (grid.to_crs(4326)[["R", "C", "cell_id", "geometry"]],
            grid_r, grid_c, land_mask)


def grid_transform():
    """Albers geo-transform of the CELL_KM grid: (x0, y0, step, grid_r, grid_c).

    The same lattice ``build_grid_cells`` lays down, returned as numbers so
    points (e.g. GLM flashes) can be binned to cells by integer division
    instead of an expensive point-in-polygon join. ``x0``/``y0`` are the SW
    corner in EPSG:5070 metres; a point at Albers (x, y) falls in
    ``C = floor((x-x0)/step)``, ``R = grid_r-1 - floor((y-y0)/step)`` (north-up).
    """
    import geopandas as gpd
    import numpy as np

    if not STATES_GEOJSON.exists():
        build_grid_cells()                            # triggers the one-time download
    states = gpd.read_file(STATES_GEOJSON)
    land = states[~states["name"].isin(_NON_CONUS)].to_crs(_ALBERS)
    step = CELL_KM * 1000.0
    minx, miny, maxx, maxy = land.total_bounds
    x0 = float(np.floor(minx / step) * step)
    y0 = float(np.floor(miny / step) * step)
    grid_c = len(np.arange(x0, maxx + step, step))
    grid_r = len(np.arange(y0, maxy + step, step))
    return x0, y0, step, grid_r, grid_c
