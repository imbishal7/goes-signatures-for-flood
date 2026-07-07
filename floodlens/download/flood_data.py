"""
NWS flood-warning puller (Flash Flood + Areal Flood) for CONUS, with polygons.

Source: Iowa Environmental Mesonet (IEM) VTEC web services — public, no key.

Two endpoints are combined:

  1. events-by-state (metadata, no geometry)
     /json/vtec_events_bystate.py?state={ST}&year={Y}&phenomena={PH}&significance=W
     → one row per warning: wfo, eventid (ETN), issue/expire, area, locations.

  2. per-event polygon (geometry, sparse metadata)
     /geojson/vtec_event.py?wfo=K{WFO}&phenomena={PH}&significance=W&etn={ETN}
        &year={Y}&sbw={0|1}&lsrs=0
     → FeatureCollection in EPSG:4326. sbw=1 is the forecaster-drawn
       storm-based warning polygon (preferred); sbw=0 is the county/zone
       aggregate (fallback when no SBW polygon exists).

Phenomena pulled (significance "W" = Warning):
  FF = Flash Flood Warning
  FA = (Areal) Flood Warning

A single VTEC event can appear in several states' lists (its polygon crosses
state lines), so events are de-duplicated by (year, wfo, phenomena, etn) and the
per-state metadata is merged.

Output: a GeoParquet of every unique warning with its polygon — a flood
ground-truth layer to pair with the GOES imagery. Written under the repo's data/
dir. Runs are resumable: the event list and each fetched polygon are cached, so
re-running only fetches what is missing.

This script is the single entry point for all flood ground truth:

  groundsource  observed flood *extents* (polygons) from Zenodo  -> data/raw/
  warnings      NWS FF/FA warning polygons from IEM              -> data/flood_warnings/
  storms        NCEI Storm Events flood reports (points)         -> data/storm_events/
  all           everything above

The ``storms`` subcommand pulls the NCEI Storm Events Database bulk CSVs
(https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/): the per-year
*details* table (EVENT_TYPE, local begin/end times + timezone, damages, flood
cause, narrative) joined with the *locations* table (precise per-event points;
BEGIN_LAT/LON fallback). Flood-related event types only; times converted to
UTC (Storm Events records fixed standard local time, e.g. "CST-6"). The
``c{stamp}`` in NCEI file names changes monthly as data is revised, so the
directory listing is scraped for the newest stamp; raw .csv.gz files are
cached under data/storm_events/raw/ and only re-downloaded on a new stamp.

All subcommands are idempotent / resumable.

CLI:
  python -m floodlens.download.flood_data warnings [--start-year ... --workers ...]
  python -m floodlens.download.flood_data groundsource [--dest PATH --force]
  python -m floodlens.download.flood_data storms [--start-year ... --types ...]
  python -m floodlens.download.flood_data all [--start-year ... --end-year ...]
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from shapely.ops import unary_union
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE = "https://mesonet.agron.iastate.edu"
LIST_URL = BASE + "/json/vtec_events_bystate.py"
GEOJSON_URL = BASE + "/geojson/vtec_event.py"

# Lower-48 + DC (CONUS); all these WFOs are K-prefixed.
CONUS_STATES = [
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
    "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
]

PHENOMENA = ["FF", "FA"]          # Flash Flood, (Areal) Flood
SIGNIFICANCE = "W"                # Warning

DEFAULT_START_YEAR = 2019
DEFAULT_END_YEAR = 2026
# Small derived dataset → keep it in the repo's data/ dir (gitignored).
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "flood_warnings"
DEFAULT_WORKERS = 12              # polite to IEM; override with --workers

_RETRIES = 4
_BACKOFF = 1.5                    # seconds, exponential

# Groundsource flood-extent dataset (Zenodo) — the companion ground-truth layer.
ZENODO_RECORD = "18647054"
GROUNDSOURCE_FILE = "groundsource_2026.parquet"
GROUNDSOURCE_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{GROUNDSOURCE_FILE}/content"
)
DEFAULT_GROUNDSOURCE_DEST = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / GROUNDSOURCE_FILE
)

# NCEI Storm Events Database bulk CSVs (public, no key).
NCEI_BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
NCEI_FILE_RE = re.compile(
    r"StormEvents_(details|locations)-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz"
)

# Flood-related EVENT_TYPE values to keep (full vocabulary is much larger).
FLOOD_EVENT_TYPES = ("Flash Flood", "Flood", "Heavy Rain", "Debris Flow")

DEFAULT_STORM_OUT_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "storm_events"
)

# Clip boxes (lon_min, lon_max, lat_min, lat_max); None = keep everything.
BBOXES: dict[str, tuple[float, float, float, float] | None] = {
    "conus": (-125.0, -66.5, 24.0, 50.0),
    "full": None,
}

STORM_DETAIL_COLS = [
    "EPISODE_ID", "EVENT_ID", "STATE", "EVENT_TYPE", "CZ_TYPE", "CZ_NAME",
    "WFO", "BEGIN_DATE_TIME", "CZ_TIMEZONE", "END_DATE_TIME",
    "INJURIES_DIRECT", "DEATHS_DIRECT", "DAMAGE_PROPERTY", "DAMAGE_CROPS",
    "SOURCE", "FLOOD_CAUSE", "BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON",
    "EVENT_NARRATIVE",
]


# ---------------------------------------------------------------------------
# HTTP with retries
# ---------------------------------------------------------------------------

def _get_json(url: str, timeout: int = 90) -> dict | None:
    """GET a URL and parse JSON, retrying with exponential backoff."""
    for attempt in range(_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == _RETRIES - 1:
                return None
            time.sleep(_BACKOFF * (2 ** attempt))
    return None


def _get_bytes(url: str, timeout: int = 120) -> bytes:
    """GET a URL and return raw bytes, retrying with exponential backoff."""
    for attempt in range(_RETRIES):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "goes-signatures-for-flood"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == _RETRIES - 1:
                raise
            time.sleep(_BACKOFF * (2 ** attempt))
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Phase 1 — event metadata (per state / year / phenomena)
# ---------------------------------------------------------------------------

def fetch_event_list(state: str, year: int, phenomena: str) -> list[dict]:
    """Return the warning-event metadata rows for one state/year/phenomena."""
    url = (
        f"{LIST_URL}?wfo=&state={state}&year={year}"
        f"&phenomena={phenomena}&significance={SIGNIFICANCE}"
    )
    data = _get_json(url)
    if not data:
        return []
    return data.get("events", [])


def _event_key(year: int, wfo: str, phenomena: str, etn: int) -> str:
    return f"{year}_{wfo}_{phenomena}_{etn}"


def gather_events(
    years: list[int], states: list[str], phenomena: list[str], workers: int
) -> dict[str, dict]:
    """Collect every unique warning across states/years, merging per-state rows.

    Keyed by (year, wfo, phenomena, etn). `states` records which states each
    event touched; `locations` unions the per-state county lists.
    """
    tasks = [(st, y, ph) for y in years for ph in phenomena for st in states]
    events: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(fetch_event_list, st, y, ph): (st, y, ph)
            for st, y, ph in tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="event lists", unit="query"):
            st, y, _ph = futures[fut]
            for e in fut.result():
                key = _event_key(y, e["wfo"], e["phenomena"], e["eventid"])
                rec = events.get(key)
                if rec is None:
                    events[key] = {
                        "key": key,
                        "year": y,
                        "wfo": e["wfo"],
                        "phenomena": e["phenomena"],
                        "significance": e["significance"],
                        "ph_name": e.get("ph_name"),
                        "sig_name": e.get("sig_name"),
                        "eventid": e["eventid"],
                        "issue": e.get("issue"),
                        "expire": e.get("expire"),
                        "product_issue": e.get("product_issue"),
                        "init_expire": e.get("init_expire"),
                        "area_iem": e.get("area"),
                        "locations": e.get("locations") or "",
                        "uri": e.get("uri"),
                        "states": {st},
                    }
                else:
                    rec["states"].add(st)
                    loc = e.get("locations") or ""
                    if loc and loc not in rec["locations"]:
                        rec["locations"] = (
                            f"{rec['locations']}; {loc}"
                            if rec["locations"] else loc
                        )
                    rec["area_iem"] = max(
                        rec["area_iem"] or 0, e.get("area") or 0
                    )
    return events


# ---------------------------------------------------------------------------
# Phase 2 — per-event polygon
# ---------------------------------------------------------------------------

def fetch_polygon(year: int, wfo: str, phenomena: str, etn: int) -> dict:
    """Fetch one event's polygon, preferring the SBW polygon over county geom.

    Returns {'wkt': str|None, 'sbw': int|None, 'n_features': int}.
    """
    for sbw in (1, 0):            # storm-based first, county/zone fallback
        url = (
            f"{GEOJSON_URL}?wfo=K{wfo}&phenomena={phenomena}"
            f"&significance={SIGNIFICANCE}&etn={etn}&year={year}"
            f"&sbw={sbw}&lsrs=0"
        )
        data = _get_json(url)
        feats = (data or {}).get("features") or []
        geoms = [shape(f["geometry"]) for f in feats if f.get("geometry")]
        if geoms:
            geom = unary_union(geoms)
            return {"wkt": geom.wkt, "sbw": sbw, "n_features": len(geoms)}
    return {"wkt": None, "sbw": None, "n_features": 0}


def _load_geom_cache(path: Path) -> dict[str, dict]:
    """Load the resumable polygon cache (jsonl keyed by event key)."""
    cache: dict[str, dict] = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    cache[rec["key"]] = rec
    return cache


def fetch_polygons(
    events: dict[str, dict], cache_path: Path, workers: int
) -> dict[str, dict]:
    """Fetch polygons for all events, skipping any already cached (resumable)."""
    cache = _load_geom_cache(cache_path)
    todo = [k for k in events if k not in cache]
    print(f"{len(events)} events; {len(cache)} cached; fetching {len(todo)}.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not todo:
        return cache

    def _work(key: str) -> dict:
        e = events[key]
        res = fetch_polygon(e["year"], e["wfo"], e["phenomena"], e["eventid"])
        res["key"] = key
        return res

    # Append to the cache file as results arrive so a kill mid-run is resumable.
    with cache_path.open("a") as out, \
            ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_work, k): k for k in todo}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="polygons", unit="event"):
            rec = fut.result()
            cache[rec["key"]] = rec
            out.write(json.dumps(rec) + "\n")
            out.flush()
    return cache


# ---------------------------------------------------------------------------
# Phase 3 — assemble & write
# ---------------------------------------------------------------------------

def build_geodataframe(
    events: dict[str, dict], geom_cache: dict[str, dict]
) -> gpd.GeoDataFrame:
    """Join metadata + polygons into a single EPSG:4326 GeoDataFrame."""
    from shapely import wkt as shapely_wkt

    rows, geoms = [], []
    for key, e in events.items():
        g = geom_cache.get(key, {})
        wkt_str = g.get("wkt")
        rows.append({
            "key": key,
            "year": e["year"],
            "wfo": e["wfo"],
            "phenomena": e["phenomena"],
            "significance": e["significance"],
            "ph_name": e["ph_name"],
            "sig_name": e["sig_name"],
            "eventid": e["eventid"],
            "issue": e["issue"],
            "expire": e["expire"],
            "issue_date": (e["issue"] or "")[:10],
            "area_iem": e["area_iem"],
            "locations": e["locations"],
            "states": ";".join(sorted(e["states"])),
            "polygon_source": (
                "sbw" if g.get("sbw") == 1
                else "county" if g.get("sbw") == 0 else "none"
            ),
            "uri": e["uri"],
        })
        geoms.append(shapely_wkt.loads(wkt_str) if wkt_str else None)

    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    return gdf.sort_values(["issue", "wfo", "eventid"]).reset_index(drop=True)


def write_outputs(gdf: gpd.GeoDataFrame, out_dir: Path) -> None:
    """Write the warnings dataset as a single GeoParquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "flood_warnings_conus.parquet"
    gdf.to_parquet(p)
    print(f"wrote {p}  ({len(gdf):,} rows)")


# ---------------------------------------------------------------------------
# Warnings orchestration
# ---------------------------------------------------------------------------

def run_warnings(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    states: list[str] | None = None,
    phenomena: list[str] | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    workers: int = DEFAULT_WORKERS,
) -> gpd.GeoDataFrame:
    """Pull all FF/FA warnings for the year range and write the dataset."""
    years = list(range(start_year, end_year + 1))
    states = states or CONUS_STATES
    phenomena = phenomena or PHENOMENA

    print(f"Years {start_year}-{end_year} | {len(states)} states | "
          f"phenomena {phenomena} | workers {workers}")

    events = gather_events(years, states, phenomena, workers)
    geom_cache = fetch_polygons(events, out_dir / "_polygon_cache.jsonl", workers)
    gdf = build_geodataframe(events, geom_cache)

    n_poly = int(gdf.geometry.notna().sum())
    by_ph = gdf.phenomena.value_counts().to_dict()
    print(f"\n{len(gdf):,} unique warnings  |  with polygon: {n_poly:,}  "
          f"|  by phenomena: {by_ph}")
    write_outputs(gdf, out_dir)
    return gdf


# ---------------------------------------------------------------------------
# Groundsource flood-extent dataset (Zenodo download)
# ---------------------------------------------------------------------------

def download_groundsource(
    dest: Path = DEFAULT_GROUNDSOURCE_DEST, force: bool = False
) -> Path:
    """Download the groundsource flood-events parquet from Zenodo (~637 MB).

    Idempotent: skips if the file already exists unless ``force``. Streams to a
    ``.part`` file and renames on success, so a killed download never leaves a
    truncated parquet behind.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"{dest} already exists ({dest.stat().st_size:,} bytes); "
              "skipping (use --force to re-download).")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading groundsource from Zenodo record {ZENODO_RECORD} -> {dest}")
    req = urllib.request.Request(
        GROUNDSOURCE_URL, headers={"User-Agent": "goes-signatures-for-flood"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total or None, unit="B", unit_scale=True, unit_divisor=1024,
            desc=GROUNDSOURCE_FILE,
        ) as bar:
            while True:
                chunk = resp.read(1 << 20)        # 1 MiB
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)
    print(f"done: {dest} ({dest.stat().st_size:,} bytes)")
    return dest


# ---------------------------------------------------------------------------
# NCEI Storm Events — remote listing + raw-file cache
# ---------------------------------------------------------------------------

def _list_ncei_files() -> dict[tuple[str, int], str]:
    """(table, year) -> newest filename, from the NCEI directory listing."""
    html = _get_bytes(NCEI_BASE_URL).decode("utf-8", errors="replace")
    newest: dict[tuple[str, int], str] = {}
    for m in NCEI_FILE_RE.finditer(html):
        table, year, stamp = m.group(1), int(m.group(2)), m.group(3)
        key = (table, year)
        if key not in newest or stamp > NCEI_FILE_RE.match(newest[key]).group(3):
            newest[key] = m.group(0)
    return newest


def _download_ncei_raw(
    remote: dict[tuple[str, int], str],
    years: list[int],
    raw_dir: Path,
    workers: int,
) -> dict[tuple[str, int], Path]:
    """Fetch details+locations files for the years (cached, parallel)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    wanted = {
        (table, y): remote[(table, y)]
        for y in years for table in ("details", "locations")
        if (table, y) in remote
    }
    todo = {k: v for k, v in wanted.items() if not (raw_dir / v).exists()}
    print(f"{len(wanted)} files | {len(wanted) - len(todo)} cached "
          f"| downloading {len(todo)}")

    def _fetch(key: tuple[str, int]) -> None:
        name = wanted[key]
        tmp = raw_dir / (name + ".part")
        tmp.write_bytes(_get_bytes(NCEI_BASE_URL + name))
        tmp.replace(raw_dir / name)

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_fetch, k): k for k in todo}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="download", unit="file"):
                fut.result()
    return {k: raw_dir / v for k, v in wanted.items()}


# ---------------------------------------------------------------------------
# NCEI Storm Events — parsing
# ---------------------------------------------------------------------------

def _to_utc(local: pd.Series, tz: pd.Series) -> pd.Series:
    """'28-APR-19 14:45:00' in zone 'CST-6' -> tz-naive UTC timestamps."""
    t = pd.to_datetime(local, format="%d-%b-%y %H:%M:%S")
    offset = (tz.str.extract(r"(-?\d+)\s*$")[0].astype("float")
              .fillna(0.0))                      # zones are standard-time fixed
    return t - pd.to_timedelta(offset, unit="h")


def _damage_usd(s: pd.Series) -> pd.Series:
    """'10.00K' / '2.5M' / '1.2B' / '' -> dollars (NaN when absent)."""
    ext = s.fillna("").str.strip().str.extract(r"^([\d.]+)([KMBkmb])?$")
    mult = ext[1].str.upper().map({"K": 1e3, "M": 1e6, "B": 1e9}).fillna(1.0)
    return pd.to_numeric(ext[0], errors="coerce") * mult


def _load_storm_year(
    year: int,
    paths: dict[tuple[str, int], Path],
    types: tuple[str, ...],
) -> pd.DataFrame | None:
    """Flood events for one year, one row per (event, point).

    An event's points are the locations-table entries (point_index 1..N,
    geom_source "locations") plus its BEGIN (point_index 0, "begin_latlon")
    and END (point_index -1, "end_latlon") coordinates — deduplicated on
    ~10 m-rounded coords, locations winning. Together they delineate the
    event's spatial span (BEGIN/END differ for ~99.9% of flood events).
    """
    if ("details", year) not in paths:
        return None
    d = pd.read_csv(paths[("details", year)], usecols=STORM_DETAIL_COLS,
                    low_memory=False)
    d = d[d["EVENT_TYPE"].isin(types)].copy()
    if d.empty:
        return None
    d["begin_utc"] = _to_utc(d["BEGIN_DATE_TIME"], d["CZ_TIMEZONE"])
    d["end_utc"] = _to_utc(d["END_DATE_TIME"], d["CZ_TIMEZONE"])
    d["damage_property_usd"] = _damage_usd(d["DAMAGE_PROPERTY"])
    d["damage_crops_usd"] = _damage_usd(d["DAMAGE_CROPS"])

    pieces: list[pd.DataFrame] = []
    if ("locations", year) in paths:
        locs = pd.read_csv(
            paths[("locations", year)],
            usecols=["EVENT_ID", "LOCATION_INDEX", "LATITUDE", "LONGITUDE"],
        ).rename(columns={"LOCATION_INDEX": "point_index",
                          "LATITUDE": "lat", "LONGITUDE": "lon"})
        locs = locs[locs["EVENT_ID"].isin(d["EVENT_ID"])].dropna(
            subset=["lat", "lon"])
        locs["geom_source"] = "locations"
        pieces.append(locs)
    for src, lat_col, lon_col, idx in (
        ("begin_latlon", "BEGIN_LAT", "BEGIN_LON", 0),
        ("end_latlon", "END_LAT", "END_LON", -1),
    ):
        p = (d[["EVENT_ID", lat_col, lon_col]]
             .rename(columns={lat_col: "lat", lon_col: "lon"})
             .dropna(subset=["lat", "lon"]))
        p["point_index"] = idx
        p["geom_source"] = src
        pieces.append(p)
    pts = pd.concat(pieces, ignore_index=True)
    pts["lat4"], pts["lon4"] = pts["lat"].round(4), pts["lon"].round(4)
    pts = (pts.drop_duplicates(["EVENT_ID", "lat4", "lon4"])   # locations win
              .drop(columns=["lat4", "lon4"]))

    m = d.merge(pts, on="EVENT_ID", how="left")    # left: keep point-less events
    m["geom_source"] = m["geom_source"].fillna("none")

    return pd.DataFrame({
        "event_id": m["EVENT_ID"],
        "episode_id": m["EPISODE_ID"],
        "event_type": m["EVENT_TYPE"],
        "flood_cause": m["FLOOD_CAUSE"],
        "state": m["STATE"],
        "cz_type": m["CZ_TYPE"],
        "cz_name": m["CZ_NAME"],
        "wfo": m["WFO"],
        "begin_utc": m["begin_utc"],
        "end_utc": m["end_utc"],
        "lat": m["lat"].astype("float32"),
        "lon": m["lon"].astype("float32"),
        "point_index": m["point_index"].fillna(0).astype("int16"),
        "geom_source": m["geom_source"],
        "injuries_direct": m["INJURIES_DIRECT"].astype("int32"),
        "deaths_direct": m["DEATHS_DIRECT"].astype("int32"),
        "damage_property_usd": m["damage_property_usd"],
        "damage_crops_usd": m["damage_crops_usd"],
        "report_source": m["SOURCE"],
        "narrative": m["EVENT_NARRATIVE"],
    })


# ---------------------------------------------------------------------------
# Storm Events orchestration
# ---------------------------------------------------------------------------

def run_storms(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    types: tuple[str, ...] = FLOOD_EVENT_TYPES,
    bbox_name: str = "conus",
    out_dir: Path = DEFAULT_STORM_OUT_DIR,
    workers: int = DEFAULT_WORKERS,
) -> gpd.GeoDataFrame:
    """Pull, filter, and write the flood storm-events point GeoParquet."""
    years = list(range(start_year, end_year + 1))
    print(f"Years {start_year}-{end_year} | types {list(types)} "
          f"| bbox {bbox_name} | workers {workers}")

    remote = _list_ncei_files()
    paths = _download_ncei_raw(remote, years, out_dir / "raw", workers)

    frames = [f for y in years
              if (f := _load_storm_year(y, paths, types)) is not None]
    df = pd.concat(frames, ignore_index=True)

    bbox = BBOXES[bbox_name]
    if bbox is not None:
        lon_min, lon_max, lat_min, lat_max = bbox
        keep = (df["lon"].between(lon_min, lon_max)
                & df["lat"].between(lat_min, lat_max))
        df = df[keep | df["lat"].isna()].reset_index(drop=True)

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    ).sort_values(["begin_utc", "event_id", "point_index"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "storm_events_flood.parquet"
    gdf.to_parquet(out_path)
    n_events = gdf["event_id"].nunique()
    by_type = gdf.drop_duplicates("event_id")["event_type"].value_counts()
    print(f"\nwrote {out_path}")
    print(f"{len(gdf):,} point rows | {n_events:,} events "
          f"| by type: {by_type.to_dict()}")
    print(f"geom_source: {gdf.geom_source.value_counts().to_dict()}")
    return gdf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download flood ground truth: NWS FF/FA warning polygons "
                    "(IEM) or the groundsource flood-extent dataset (Zenodo).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # -- warnings subcommand (IEM FF/FA pull) --------------------------------
    w = sub.add_parser(
        "warnings", help="Pull CONUS NWS Flash Flood + Flood warning polygons."
    )
    w.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    w.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    w.add_argument(
        "--states", nargs="+", default=None, metavar="ST",
        help=f"State codes (default: all {len(CONUS_STATES)} CONUS states).",
    )
    w.add_argument(
        "--phenomena", nargs="+", default=None, choices=["FF", "FA"],
        help="Phenomena to pull (default: FF FA).",
    )
    w.add_argument("--out", default=str(DEFAULT_OUT_DIR), metavar="DIR")
    w.add_argument("--workers", type=int, default=DEFAULT_WORKERS, metavar="N")

    # -- groundsource subcommand (Zenodo download) ---------------------------
    g = sub.add_parser(
        "groundsource",
        help="Download the groundsource flood-extent parquet from Zenodo.",
    )
    g.add_argument(
        "--dest", default=str(DEFAULT_GROUNDSOURCE_DEST), metavar="PATH",
        help=f"Output path (default: {DEFAULT_GROUNDSOURCE_DEST}).",
    )
    g.add_argument(
        "--force", action="store_true", help="Re-download even if present."
    )

    # -- storms subcommand (NCEI Storm Events pull) --------------------------
    s = sub.add_parser(
        "storms",
        help="Pull NCEI Storm Events flood reports (points) into a GeoParquet.",
    )
    s.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    s.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    s.add_argument(
        "--types", nargs="+", default=list(FLOOD_EVENT_TYPES), metavar="TYPE",
        help=f"EVENT_TYPE values to keep (default: {list(FLOOD_EVENT_TYPES)}).",
    )
    s.add_argument("--bbox", default="conus", choices=list(BBOXES),
                   help="Spatial clip (default: conus).")
    s.add_argument("--out", default=str(DEFAULT_STORM_OUT_DIR), metavar="DIR")
    s.add_argument("--workers", type=int, default=DEFAULT_WORKERS, metavar="N")

    # -- all subcommand (everything) ------------------------------------------
    a = sub.add_parser(
        "all", help="Download all three layers: groundsource, warnings, storms."
    )
    a.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    a.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    a.add_argument("--workers", type=int, default=DEFAULT_WORKERS, metavar="N")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "warnings":
        run_warnings(
            start_year=args.start_year,
            end_year=args.end_year,
            states=args.states,
            phenomena=args.phenomena,
            out_dir=Path(args.out),
            workers=args.workers,
        )
    elif args.command == "groundsource":
        download_groundsource(dest=Path(args.dest), force=args.force)
    elif args.command == "storms":
        run_storms(
            start_year=args.start_year,
            end_year=args.end_year,
            types=tuple(args.types),
            bbox_name=args.bbox,
            out_dir=Path(args.out),
            workers=args.workers,
        )
    elif args.command == "all":
        print("=" * 70)
        print("1/3  groundsource flood extents (Zenodo) -> data/raw/")
        print("=" * 70)
        download_groundsource()
        print("=" * 70)
        print("2/3  NWS FF/FA warnings (IEM) -> data/flood_warnings/")
        print("=" * 70)
        run_warnings(start_year=args.start_year, end_year=args.end_year,
                     workers=args.workers)
        print("=" * 70)
        print("3/3  NCEI Storm Events -> data/storm_events/")
        print("=" * 70)
        run_storms(start_year=args.start_year, end_year=args.end_year,
                   workers=args.workers)


if __name__ == "__main__":
    main()
