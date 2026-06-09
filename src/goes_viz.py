"""
GOES imagery viewer + flood overlay for the cloud → flood exploration.

This module powers ``notebooks/flood_clouds.ipynb``. It reuses the reprojection /
web-map machinery first prototyped inline in ``notebooks/preview_goes.ipynb`` and
adds the two new layers the flood question needs:

  * a 25 x 25 km reference grid clipped to the CONUS land area, and
  * the next day's flood polygons from the ground-truth GeoParquet,

so a single GOES time-lapse (6 daytime frames) can be watched against the floods
it may have produced the following day.

Everything renders on one scroll-zoom folium map; the grid and flood layers are
toggleable via the layer control. Imagery is reprojected geostationary → EPSG:4326
so it aligns with the basemap and the (already-4326) flood polygons.

Band reference (every MCMIPC file holds all 16):
  true color = RGB (2, 3, 1);  band 13 = clean IR (cold cloud tops);  band 8 = vapor
"""

import base64
import io
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from branca.element import MacroElement
from folium.raster_layers import ImageOverlay
from jinja2 import Template
from PIL import Image
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("/mnt/disk1/goes-data")
AUX_DIR = DATA_DIR / "aux"                       # cached boundaries / grids
# Resolve the ground-truth parquet relative to the repo, not the caller's cwd.
_REPO = Path(__file__).resolve().parent.parent
FLOOD_PARQUET = _REPO / "data/raw/groundsource_2026.parquet"

# CONUS land box (lon_min, lon_max, lat_min, lat_max) — drops AK/HI/PR/territories.
CONUS_BBOX = (-125.0, -66.5, 24.0, 50.0)

# US states GeoJSON (low-res, cached on disk1); used to clip the grid to land.
US_STATES_GEOJSON = AUX_DIR / "us-states.geojson"
US_STATES_URL = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
    "master/data/geojson/us-states.json"
)
NON_CONUS_STATES = {"Alaska", "Hawaii", "Puerto Rico"}

CONUS_ALBERS = 5070                              # EPSG, equal-area metres for the grid

_BLANK = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAA"
          "C0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC")  # 1x1 transparent


# ---------------------------------------------------------------------------
# File discovery & selection
# ---------------------------------------------------------------------------

def _scan_token(p: Path) -> str:
    """Return the filename's ``s{YYYYDDDHHMMSSf}`` scan-start token."""
    for part in p.name.split("_"):
        if part.startswith("s") and part[1:].isdigit():
            return part
    return p.name


def scan_time(p: Path) -> str:
    """``HH:MM UTC`` from a filename's scan-start token."""
    t = _scan_token(p)
    if t.startswith("s") and len(t) >= 12:
        return f"{t[8:10]}:{t[10:12]} UTC"
    return "??:??"


def _stamp(p: Path) -> str:
    """``YYYY-MM-DD HH:MM UTC`` from a filename's scan-start token."""
    t = _scan_token(p)
    if t.startswith("s") and len(t) >= 12:
        d = datetime.strptime(t[1:8], "%Y%j")
        return f"{d:%Y-%m-%d} {t[8:10]}:{t[10:12]} UTC"
    return p.name


def find_files(dt: date, data_dir: Path = DATA_DIR) -> list[Path]:
    """All GOES NetCDFs for a date, sorted by scan time (handles both satellites)."""
    pat = f"*/{dt.year}/{dt.month:02d}/{dt.day:02d}/*.nc"
    return sorted(data_dir.glob(pat), key=_scan_token)


def open_goes(dt: date, file_index: int = 0, data_dir: Path = DATA_DIR) -> xr.Dataset:
    """List a date's files and open one (default the first)."""
    files = find_files(dt, data_dir)
    if not files:
        raise FileNotFoundError(f"No .nc files for {dt} under {data_dir}")
    for i, p in enumerate(files):
        arrow = "->" if i == file_index else "  "
        print(f"{arrow} [{i}] {scan_time(p)}  {p.name}")
    return xr.open_dataset(files[file_index], decode_times=False)


# ---------------------------------------------------------------------------
# Band access & scaling
# ---------------------------------------------------------------------------

def cmi(ds: xr.Dataset, n: int) -> xr.DataArray:
    """The ``CMI_C<n>`` band (decoded reflectance or brightness temperature)."""
    return ds[f"CMI_C{n:02d}"]


def _is_bt(da: xr.DataArray) -> bool:
    return str(da.attrs.get("units", "")).strip().upper().startswith("K")


def _cmap(da: xr.DataArray) -> str:
    return "gray_r" if _is_bt(da) else "gray"    # IR: cold cloud tops = white


def _stretch(a, vmin=None, vmax=None) -> tuple[float, float]:
    lo = np.nanpercentile(a, 2) if vmin is None else vmin
    hi = np.nanpercentile(a, 98) if vmax is None else vmax
    return float(lo), float(hi)


def _norm(a, lo, hi):
    return np.clip((a - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(a)


# ---------------------------------------------------------------------------
# Crop on the native geostationary grid
# ---------------------------------------------------------------------------

def _geos(ds: xr.Dataset) -> pyproj.CRS:
    return pyproj.CRS.from_cf(dict(ds["goes_imager_projection"].attrs))


def crop_lonlat(ds: xr.Dataset, lon_min, lon_max, lat_min, lat_max) -> xr.Dataset:
    """Subset to a lon/lat box (degrees) before reprojecting — faster + zoomed in."""
    h = ds["goes_imager_projection"].attrs["perspective_point_height"]
    tf = pyproj.Transformer.from_crs("EPSG:4326", _geos(ds), always_xy=True)
    lons = [lon_min, lon_max, lon_min, lon_max]
    lats = [lat_min, lat_min, lat_max, lat_max]
    xs, ys = tf.transform(lons, lats)
    xs = np.array(xs) / h
    ys = np.array(ys) / h
    xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
    if xs.size == 0 or ys.size == 0:
        raise ValueError("box is off the Earth disk for this satellite")
    return ds.sel(x=slice(xs.min(), xs.max()), y=slice(ys.max(), ys.min()))


# ---------------------------------------------------------------------------
# Reproject geostationary → lat/lon, then encode an RGBA web-map overlay
# ---------------------------------------------------------------------------

def reproject_bands(ds: xr.Dataset, bands: list[int]):
    """Reproject the given bands to EPSG:4326.

    Returns ``(DataArray[band, y, x], bounds=[[south, west], [north, east]])``,
    north-up / west-left so it drops straight onto a Leaflet overlay.
    """
    h = ds["goes_imager_projection"].attrs["perspective_point_height"]
    da = xr.concat([cmi(ds, n).reset_coords(drop=True) for n in bands], dim="band")
    da = da.assign_coords(x=ds["x"] * h, y=ds["y"] * h)
    da = da.rio.write_crs(_geos(ds)).rio.set_spatial_dims(x_dim="x", y_dim="y")
    da = da.rio.reproject("EPSG:4326", nodata=np.nan)
    da = da.sortby("y", ascending=False).sortby("x")
    minx, miny, maxx, maxy = da.rio.bounds()
    return da, [[miny, minx], [maxy, maxx]]


def _rgba_single(a, cmap, lo, hi):
    finite = np.isfinite(a)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(lo, hi), cmap=cmap)
    rgba = sm.to_rgba(np.nan_to_num(a, nan=lo), bytes=True)
    rgba[..., 3] = np.where(finite, 255, 0).astype("uint8")
    return rgba


def _rgba_rgb(arr3, gamma):
    chans, alpha = [], np.ones(arr3.shape[1:], dtype=bool)
    for a in arr3:
        chans.append(_norm(a, *_stretch(a)))
        alpha = alpha & np.isfinite(a)
    rgb = np.nan_to_num(np.clip(np.dstack(chans) ** (1 / gamma), 0, 1))
    a8 = np.where(alpha, 255, 0).astype("uint8")
    return np.dstack([(rgb * 255).astype("uint8"), a8])


def _data_uri(rgba) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _downsample(rgba, max_px: int):
    """Shrink an RGBA frame so the embedded HTML stays small."""
    h, w = rgba.shape[:2]
    if max(h, w) <= max_px:
        return rgba
    s = max_px / max(h, w)
    im = Image.fromarray(rgba, "RGBA").resize(
        (max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return np.asarray(im)


def _frame_rgba(ds: xr.Dataset, band, rgb, crop):
    """Reproject one dataset to an RGBA array + lat/lon bounds for an overlay."""
    sub = crop_lonlat(ds, *crop) if crop else ds
    if rgb is not None:
        da_ll, bounds = reproject_bands(sub, list(rgb))
        return _rgba_rgb(da_ll.values, 2.2), bounds
    da_ll, bounds = reproject_bands(sub, [band])
    a = da_ll.isel(band=0).values
    return _rgba_single(a, _cmap(cmi(sub, band)), *_stretch(a)), bounds


# ---------------------------------------------------------------------------
# 25 km CONUS land grid
# ---------------------------------------------------------------------------

def conus_land(states_geojson: Path = US_STATES_GEOJSON) -> gpd.GeoDataFrame:
    """CONUS land polygon (lower-48 + DC) in EPSG:4326.

    Reads the cached low-res US-states GeoJSON, downloading it once to
    ``aux/`` if missing, and drops Alaska / Hawaii / Puerto Rico.
    """
    if not states_geojson.exists():
        import urllib.request
        states_geojson.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading US states boundary → {states_geojson}")
        urllib.request.urlretrieve(US_STATES_URL, states_geojson)
    g = gpd.read_file(states_geojson)
    return g[~g["name"].isin(NON_CONUS_STATES)].to_crs(4326)


def conus_grid(cell_km: float = 25.0, cache: bool = True) -> gpd.GeoDataFrame:
    """A fishnet of ``cell_km`` square cells covering CONUS land, in EPSG:4326.

    Cells are built in CONUS Albers (EPSG:5070) so they are genuinely
    ``cell_km`` on a side, then kept only where they intersect the land polygon.
    The result is cached under ``aux/`` so repeat calls are instant. Each row has
    a ``cell_id`` (``r{row}c{col}``) — the natural sampling unit for later feature
    extraction.
    """
    cache_path = AUX_DIR / f"conus_grid_{int(cell_km)}km.parquet"
    if cache and cache_path.exists():
        return gpd.read_parquet(cache_path)

    land = conus_land().to_crs(CONUS_ALBERS)
    land_union = land.union_all()
    step = cell_km * 1000.0
    minx, miny, maxx, maxy = land.total_bounds

    cells, ids = [], []
    ys = np.arange(np.floor(miny / step) * step, maxy + step, step)
    xs = np.arange(np.floor(minx / step) * step, maxx + step, step)
    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            cells.append(box(x0, y0, x0 + step, y0 + step))
            ids.append(f"r{r:03d}c{c:03d}")
    grid = gpd.GeoDataFrame({"cell_id": ids}, geometry=cells, crs=CONUS_ALBERS)
    grid = grid[grid.intersects(land_union)].reset_index(drop=True)
    grid = grid.to_crs(4326)

    if cache:
        AUX_DIR.mkdir(parents=True, exist_ok=True)
        grid.to_parquet(cache_path)
    return grid


# ---------------------------------------------------------------------------
# Ground-truth flood polygons
# ---------------------------------------------------------------------------

def load_floods(
    target: date,
    bbox: tuple[float, float, float, float] = CONUS_BBOX,
    parquet: Path = FLOOD_PARQUET,
) -> gpd.GeoDataFrame:
    """Flood polygons active on ``target`` within ``bbox`` (lon/lat).

    "Active" means ``start_date <= target <= end_date``. The date filter is pushed
    down to Parquet (ISO strings compare lexicographically) so only the relevant
    rows are read; the bbox is then applied spatially.
    """
    iso = target.isoformat()
    g = gpd.read_parquet(
        parquet,
        columns=["uuid", "area_km2", "start_date", "end_date", "geometry"],
        filters=[("start_date", "<=", iso), ("end_date", ">=", iso)],
    )
    lon_min, lon_max, lat_min, lat_max = bbox
    return g.cx[lon_min:lon_max, lat_min:lat_max].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Folium layers
# ---------------------------------------------------------------------------

def grid_layer(grid: gpd.GeoDataFrame, name: str = "25 km grid") -> folium.GeoJson:
    """Thin, unfilled grid outlines as a toggleable GeoJson layer."""
    return folium.GeoJson(
        grid[["cell_id", "geometry"]].to_json(),
        name=name,
        style_function=lambda _f: {
            "color": "#444", "weight": 0.4, "fill": False, "opacity": 0.5,
        },
        tooltip=folium.GeoJsonTooltip(fields=["cell_id"], aliases=["cell"]),
    )


def flood_layer(
    floods: gpd.GeoDataFrame, name: str = "next-day floods"
) -> folium.GeoJson:
    """Filled flood polygons as a toggleable GeoJson layer (area tooltip)."""
    return folium.GeoJson(
        floods[["area_km2", "start_date", "geometry"]].to_json(),
        name=name,
        style_function=lambda _f: {
            "color": "#0b4dd6", "weight": 0.6, "fillColor": "#1f78ff",
            "fillOpacity": 0.55,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["area_km2", "start_date"], aliases=["km²", "start"]
        ),
    )


def _base_map(bounds) -> folium.Map:
    """A scroll-zoom folium map with OSM / light / satellite basemaps."""
    (s, w), (n, e) = bounds
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=5,
                   tiles="OpenStreetMap")
    folium.TileLayer("CartoDB positron", name="light").add_to(m)
    folium.TileLayer(
        tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/"
               "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
        attr="Esri World Imagery", name="satellite",
    ).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Time-lapse control (bottom-left play/pause + slider over an ImageOverlay)
# ---------------------------------------------------------------------------

class _TimeLapse(MacroElement):
    """A bottom-left play/slider that cycles an ImageOverlay through frames."""

    _template = Template("""
        {% macro script(this, kwargs) %}
        (function(){
          var fr={{this.frames_json}}, lb={{this.labels_json}};
          var ov={{this.overlay_name}}, mp={{this._parent.get_name()}};
          var nm="{{this.get_name()}}", iv={{this.interval}}, i=0, t=null, pl=false;
          var c=L.control({position:'bottomleft'});
          c.onAdd=function(){
            var d=L.DomUtil.create('div','');
            d.style.cssText='background:rgba(255,255,255,.88);padding:6px 8px;'
              +'font:12px sans-serif;border-radius:4px';
            d.innerHTML='<button id="b_'+nm+'">&#9658;</button> '
              +'<input id="s_'+nm+'" type="range" min="0" max="'+(fr.length-1)
              +'" value="0" style="width:240px;vertical-align:middle"> '
              +'<span id="l_'+nm+'"></span>';
            L.DomEvent.disableClickPropagation(d);
            return d;
          };
          c.addTo(mp);
          function show(k){i=(k+fr.length)%fr.length;ov.setUrl(fr[i]);
            document.getElementById('s_'+nm).value=i;
            document.getElementById('l_'+nm).innerText=lb[i];}
          function stop(){pl=false;document.getElementById('b_'+nm).innerHTML='&#9658;';
            clearInterval(t);}
          function go(){pl=true;
            document.getElementById('b_'+nm).innerHTML='&#10074;&#10074;';
            t=setInterval(function(){show(i+1);},iv);}
          setTimeout(function(){
            show(0);
            document.getElementById('b_'+nm).onclick=function(){pl?stop():go();};
            document.getElementById('s_'+nm).oninput=function(e){stop();
              show(parseInt(e.target.value));};
            go();
          },300);
        })();
        {% endmacro %}
    """)

    def __init__(self, overlay_name, frames, labels, interval=800):
        super().__init__()
        self._name = "TimeLapse"
        self.overlay_name = overlay_name
        self.frames_json = json.dumps(frames)
        self.labels_json = json.dumps(labels)
        self.interval = interval


# ---------------------------------------------------------------------------
# The combined view: clouds (time-lapse) + 25 km grid + next-day floods
# ---------------------------------------------------------------------------

def flood_cloud_timelapse(
    goes_date: date,
    band: int = 13,
    rgb: tuple[int, int, int] | None = None,
    crop: tuple[float, float, float, float] | None = None,
    show_grid: bool = True,
    grid_cell_km: float = 25.0,
    flood_lag_days: int = 1,
    max_px: int = 1400,
    interval: int = 800,
    opacity: float = 0.8,
    data_dir: Path = DATA_DIR,
) -> folium.Map:
    """Watch a day's GOES clouds against the floods they may have caused.

    Builds a time-lapse of every GOES frame on ``goes_date`` (6 daytime images by
    default), overlays the 25 km CONUS land grid, and draws the flood polygons
    active ``flood_lag_days`` later — all on one scroll-zoom map. The grid and
    flood layers toggle from the layer control (top-right); the play/pause +
    slider sit at the bottom-left.

    Use ``band=N`` (13 = cold cloud tops, 8 = water vapor) or ``rgb=(2, 3, 1)``
    for true colour. ``crop=(lon_min, lon_max, lat_min, lat_max)`` focuses the
    view and speeds reprojection; grid + floods are clipped to it too.
    """
    files = find_files(goes_date, data_dir)
    if not files:
        raise FileNotFoundError(f"No GOES files for {goes_date} under {data_dir}")
    print(f"{len(files)} cloud frame(s) on {goes_date}; reprojecting...")

    frames, labels, bounds = [], [], None
    for j, p in enumerate(files):
        ds = xr.open_dataset(p, decode_times=False)
        try:
            rgba, b = _frame_rgba(ds, band, rgb, crop)
        finally:
            ds.close()
        bounds = bounds or b
        frames.append(_data_uri(_downsample(rgba, max_px)))
        labels.append(_stamp(p))
        print(f"  {j + 1}/{len(files)}  {labels[-1]}", end="\r")
    print()

    # next-day floods, clipped to the view
    target = goes_date + timedelta(days=flood_lag_days)
    view_bbox = crop if crop else CONUS_BBOX
    floods = load_floods(target, bbox=view_bbox)
    print(f"{len(floods)} flood polygon(s) active on {target} "
          f"({floods.area_km2.sum():,.0f} km²)")

    m = _base_map(bounds)
    ov = ImageOverlay(_BLANK, bounds=bounds, opacity=opacity,
                      name=f"GOES clouds {goes_date}")
    ov.add_to(m)

    if show_grid:
        grid = conus_grid(grid_cell_km)
        lon_min, lon_max, lat_min, lat_max = view_bbox
        grid = grid.cx[lon_min:lon_max, lat_min:lat_max]
        grid_layer(grid, f"{int(grid_cell_km)} km grid").add_to(m)

    if len(floods):
        flood_layer(floods, f"floods {target}").add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(bounds)
    m.add_child(_TimeLapse(ov.get_name(), frames, labels, interval))
    return m
