"""GOES-pixel -> 50 km-cell index (the geographic regrid lookup).

Shared by every model trainer (`trainers/*.py`) and `model_results.ipynb`:
``build_pix2cell()`` maps each full-res ABI pixel to its flat cell id in the CELL_KM
grid (CONUS Albers), caching the result to disk. The heavy pyproj projection math lives
here, in one canonical place, so the trainers stay self-contained otherwise.
"""
import sys
from pathlib import Path

import numpy as np

# config.py (repo root) is the single source for the grid + paths
ROOT = Path(__file__).resolve().parent
while not (ROOT / "config.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from config import (  # noqa: E402
    CACHE_DIR,
    CELL_KM,
    DATA_DIR,
    build_grid_cells,
    grid_transform,
)


def build_pix2cell():
    """Full-res ABI pixel -> flat cell id, cached to disk.

    Returns (pix2cell (1500,2500) int32, grid_r, grid_c, land_mask).
    """
    import netCDF4
    import pyproj

    _, grid_r, grid_c, land_mask = build_grid_cells()
    cache = CACHE_DIR / f"pix2cell_{CELL_KM}km.npy"
    if cache.exists():
        return np.load(cache), grid_r, grid_c, land_mask

    gx0, gy0, gstep, *_ = grid_transform()
    ref = sorted(DATA_DIR.glob("*/2019/*/*/*.nc"))[0]
    with netCDF4.Dataset(ref) as nc:
        proj = nc["goes_imager_projection"]
        geos = pyproj.CRS.from_cf({k: proj.getncattr(k) for k in proj.ncattrs()})
        sat_h = float(proj.perspective_point_height)
        xc = nc["x"][:].astype(np.float64) * sat_h
        yc = nc["y"][:].astype(np.float64) * sat_h
    xx, yy = np.meshgrid(xc, yc)
    tf = pyproj.Transformer.from_crs(geos, 5070, always_xy=True)
    ax, ay = tf.transform(xx.ravel(), yy.ravel())
    ax = np.asarray(ax).reshape(xx.shape)
    ay = np.asarray(ay).reshape(xx.shape)
    col = np.floor((ax - gx0) / gstep)
    row = (grid_r - 1) - np.floor((ay - gy0) / gstep)
    ok = (np.isfinite(ax) & np.isfinite(ay)
          & (col >= 0) & (col < grid_c) & (row >= 0) & (row < grid_r))
    p2c = np.full(ax.shape, -1, np.int32)
    p2c[ok] = (row[ok] * grid_c + col[ok]).astype(np.int32)
    cache.parent.mkdir(parents=True, exist_ok=True)       # np.save won't create the dir
    np.save(cache, p2c)
    return p2c, grid_r, grid_c, land_mask
