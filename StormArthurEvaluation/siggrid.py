"""Self-contained output-grid helper for the standalone ``StormArthurEvaluation/`` experiment.

Copied from the repo-root ``config.py`` so this sub-project has no dependency on the
main pipeline — it can be run (or moved out) on its own. Defines the CELL_KM square-cell
CONUS-land grid and the constants the StormArthurEvaluation notebooks/scripts import.
"""
from pathlib import Path

SIG_ROOT = Path(__file__).resolve().parent

# ---- output grid ----------------------------------------------------------
CELL_KM = 50                                       # square output cell size (km)
STATES_GEOJSON = SIG_ROOT / "data/us-states.geojson"   # CONUS land boundary (local copy)

_STATES_URL = ("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
               "master/data/geojson/us-states.json")
_NON_CONUS = {"Alaska", "Hawaii", "Puerto Rico"}   # dropped from the land grid
_ALBERS = 5070                                     # CONUS Albers equal-area (m)


def build_grid_cells():
    """Generate the CELL_KM square-cell grid over CONUS land (north-up).

    CELL_KM-metre cells are laid in CONUS Albers (EPSG:5070) and kept where they
    intersect US land (lower-48 + DC, from STATES_GEOJSON — downloaded once if missing).

    Returns
        cells      GeoDataFrame[R, C, cell_id, geometry] in EPSG:4326 (row 0 = north,
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
    grid["cell_id"] = ("r" + grid["R"].map("{:03d}".format)
                       + "c" + grid["C"].map("{:03d}".format))

    land_mask = np.zeros((grid_r, grid_c), dtype=bool)
    land_mask[grid["R"], grid["C"]] = True
    return (grid.to_crs(4326)[["R", "C", "cell_id", "geometry"]],
            grid_r, grid_c, land_mask)
