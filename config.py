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
# Full dataset — 2019-2025 on full-day 3-hourly GOES; 2026 dropped (truncated labels).
# Splits are NOT static: notebooks/model/foldsplit.py does blocked K-fold CV (one fixed
# test set + rotating train/val folds) computed at train time. nb 01 only marks 2026 "unused".
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

# per-frame GOES: the 5 bands + a per-frame lead-time channel (hours before day D,
# from _t.npy). The 50 km feature cache (nb 01) pools these to per-cell features.
T_FRAMES = 8                                      # 3-hourly frames/day, full UTC day (00,03,..,21Z)
IMG_H, IMG_W = 1500, 2500                         # ABI CONUS 2 km grid (rows, cols)

# >>> OUTPUT GRID SIZE LEVER <<<
# Square output cells CELL_KM on a side, generated directly over CONUS land in
# an equal-area projection (see build_grid_cells). Any size works: 50, 40, 75...
CELL_KM = 50

# Single canonical cache (built by nb 01): pure per-cell 50 km GOES/GLM signature features
# (seq + daily summaries) for 2019-2025 - no full-resolution image. ~9 GB, root NVMe.
# Training hyperparameters, loss, and the CV split all live with the trainers /
# foldsplit.py, not here (each trainer is self-contained; see notebooks/model/).
CACHE_DIR = ROOT / "cache" / "goes_grid50_2019_2026"   # project NVMe (fast reads)

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
