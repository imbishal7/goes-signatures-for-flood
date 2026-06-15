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
DATA_DIR = Path("/mnt/disk1/goes-data")          # GOES ABI NetCDFs
GLM_DIR = Path("/mnt/disk1/glm-data")            # GLM lightning parquet/day
STATES_GEOJSON = DATA_DIR / "aux/us-states.geojson"       # CONUS land boundary
UNIFIED_PARQUET = ROOT / "data/flood_warnings/floods_unified.parquet"

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
YEAR = 2019

# GOES inputs: 6 ABI bands chosen to work day AND night, + 1 GLM channel
BANDS = (2, 6, 8, 10, 13, 16)
N_BAND = len(BANDS)
N_CH = N_BAND + 1                                 # bands + whole-day GLM map
T_FRAMES = 6                                      # daytime frames/day (16-21 UTC)
IMG_H, IMG_W = 1500, 2500                         # ABI CONUS 2 km grid (rows, cols)

# >>> OUTPUT GRID SIZE LEVER <<<
# Square output cells CELL_KM on a side, generated directly over CONUS land in
# an equal-area projection (see build_grid_cells). Any size works: 50, 40, 75...
CELL_KM = 50

# encoder downsample before pixel->cell pooling (notebook 02)
POOL_STRIDE = 2

# ---------------------------------------------------------------------------
# Derived artifact paths (depend on YEAR / CELL_KM so changing the grid or year
# never silently reuses a stale cache)
# ---------------------------------------------------------------------------
STATS_PATH = DATA_DIR / f"aux/band_stats_{YEAR}.json"
CACHE_DIR = ROOT / "cache" / f"floodnet_{YEAR}"   # on the project's NVMe (fast reads)
POOL_IDX_PATH = CACHE_DIR / f"pool_index_s{POOL_STRIDE}_{CELL_KM}km.npy"
CKPT_DIR = Path("/mnt/disk1/models/floodnet_convlstm")

# ---------------------------------------------------------------------------
# Train / val / test split (random, reproducible) — notebook 01
# ---------------------------------------------------------------------------
SPLIT_SEED = 0
SPLIT_FRACS = (0.70, 0.20, 0.10)                  # train / val / test

# ---------------------------------masked------------------------------------
# Training hyper-parameters — notebook 02
# ---------------------------------------------------------------------------
BATCH_PER_GPU = 1
EPOCHS = 5
LR = 3e-4
WORKERS = 8

# ---------------------------------------------------------------------------
# Loss — Tversky (Dice family) on a neighborhood-tolerant target
# ---------------------------------------------------------------------------
# alpha weights false positives, beta weights false negatives; beta > alpha
# favors recall (don't miss floods). NEIGHBOR_W gives a flooded cell's 1st-ring
# neighbours a soft target of this weight, so predicting a neighbour is partial
# credit (your 1 / 0.5 / 0 scheme). Set NEIGHBOR_W = 0 for hard 0/1 targets.
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
NEIGHBOR_W = 0.5


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
        cells      GeoDataFrame[R, C, geometry] in EPSG:4326 (row 0 = north,
                   col 0 = west — matching the GOES imagery orientation).
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

    land_mask = np.zeros((grid_r, grid_c), dtype=bool)
    land_mask[grid["R"], grid["C"]] = True
    return grid.to_crs(4326)[["R", "C", "geometry"]], grid_r, grid_c, land_mask
