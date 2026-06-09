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

Output: a GeoParquet (and/or CSV with WKT geometry) of every unique warning with
its polygon — a flood ground-truth layer to pair with the GOES imagery. Written
under the repo's data/ dir. Runs are resumable: the event list and each fetched
polygon are cached, so re-running only fetches what is missing.

This script also fetches the companion *groundsource* flood-extent dataset from
Zenodo (the ``groundsource`` subcommand) so the full flood ground truth can be
reproduced from one place.

CLI:
  python src/download_flood_data.py warnings [--start-year ... --workers ...]
  python src/download_flood_data.py groundsource [--dest PATH --force]
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
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


def write_outputs(gdf: gpd.GeoDataFrame, out_dir: Path, fmt: str) -> None:
    """Write the dataset as GeoParquet and/or CSV (geometry as WKT)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "flood_warnings_conus"
    if fmt in ("parquet", "both"):
        p = out_dir / f"{stem}.parquet"
        gdf.to_parquet(p)
        print(f"wrote {p}  ({len(gdf):,} rows)")
    if fmt in ("csv", "both"):
        c = out_dir / f"{stem}.csv"
        df = gdf.copy()
        df["geometry_wkt"] = df.geometry.apply(lambda g: g.wkt if g else "")
        df.drop(columns="geometry").to_csv(c, index=False)
        print(f"wrote {c}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    states: list[str] | None = None,
    phenomena: list[str] | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    workers: int = DEFAULT_WORKERS,
    fmt: str = "both",
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
    write_outputs(gdf, out_dir, fmt)
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
    w.add_argument("--format", default="both", choices=["parquet", "csv", "both"])

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

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "warnings":
        run(
            start_year=args.start_year,
            end_year=args.end_year,
            states=args.states,
            phenomena=args.phenomena,
            out_dir=Path(args.out),
            workers=args.workers,
            fmt=args.format,
        )
    elif args.command == "groundsource":
        download_groundsource(dest=Path(args.dest), force=args.force)


if __name__ == "__main__":
    main()
