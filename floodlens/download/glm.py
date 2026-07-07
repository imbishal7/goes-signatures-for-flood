"""
GLM-L2-LCFA flash downloader for GOES-16 / GOES-19 — consolidated per day.

The GLM (Geostationary Lightning Mapper) Level-2 LCFA product is a point dataset
of detected lightning, delivered as one small NetCDF every 20 seconds
(~4,320 files/day). This tool streams a day's files straight into memory, keeps
only the **flash** level (centroid + energy + area + timing), clips to a lon/lat
box, and writes a single per-day parquet — raw bytes never touch disk. No
gridding: full flash-point resolution is preserved and can be re-gridded later.

Hierarchy reminder: events (per-pixel, per-2ms) -> groups (per-pulse) -> flashes
(clustered). We keep flashes only; that's the meteorological unit and ~1/22 the
size of the per-pixel events.

S3 layout (anonymous, identical for both satellites):
  s3://noaa-goes{16|19}/GLM-L2-LCFA/{year}/{doy:03d}/{hour:02d}/
    OR_GLM-L2-LCFA_G{16|19}_s{...}_e{...}_c{...}.nc

Both buckets are public, so listing and GETs go over plain unsigned HTTPS via
urllib3 — much lower per-request CPU than boto3, which matters at ~4,320
requests/day across many worker processes. Each file is fetched into memory and
parsed immediately in the download thread (netCDF4 from a bytes buffer), so
download and parse overlap instead of running as two sequential phases.

Satellite cutover (same as ABI): GOES-16 through 2025-04-06, GOES-19 after.

Output columns (one row per flash):
  time_start, time_end (UTC) | lat, lon | energy (J) | area (m^2) | quality_flag
"""

import argparse
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import netCDF4
import pandas as pd
import urllib3
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SATELLITES: dict[str, str] = {"GOES16": "noaa-goes16", "GOES19": "noaa-goes19"}
PRODUCT = "GLM-L2-LCFA"

GOES16_END_DATE = date(2025, 4, 6)

# Default range matches the ABI/GOES download span (see download_goes.py).
DEFAULT_START_DATE = date(2026, 6, 1)
DEFAULT_END_DATE = date(2026, 6, 23)

# Clip boxes (lon_min, lon_max, lat_min, lat_max); None = full disk.
CLIP_BOXES: dict[str, tuple[float, float, float, float] | None] = {
    "margin": (-130.0, -61.0, 19.0, 55.0),   # CONUS + ~5 deg (default)
    "conus": (-125.0, -66.0, 24.0, 50.0),
    "full": None,
}

# Flash fields kept (source name -> output column).
FLASH_FIELDS: dict[str, str] = {
    "flash_time_offset_of_first_event": "time_start",
    "flash_time_offset_of_last_event": "time_end",
    "flash_lat": "lat",
    "flash_lon": "lon",
    "flash_energy": "energy",
    "flash_area": "area",
    "flash_quality_flag": "quality_flag",
}

DEFAULT_OUT_DIR = Path("/mnt/disk4/recent-glm")
DEFAULT_WORKERS = 32              # fetch+parse threads per day (network-bound)
DEFAULT_DAY_WORKERS = 8          # days processed in parallel (range mode)

_S3_XMLNS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# netCDF4/HDF5 C libraries are not thread-safe; parsing is serialized behind
# this lock (a tiny LCFA file parses in ~2 ms, so contention is negligible
# next to the ~50 ms network fetch).
_NC_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Unsigned HTTP access to the public buckets
# ---------------------------------------------------------------------------

def _make_http(workers: int) -> urllib3.PoolManager:
    """Connection pool with retry/backoff (S3 returns 503 SlowDown under load)."""
    retry = urllib3.util.Retry(
        total=5, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    return urllib3.PoolManager(maxsize=workers, retries=retry)


def _get(http: urllib3.PoolManager, bucket: str, key: str) -> bytes:
    resp = http.request("GET", f"https://{bucket}.s3.amazonaws.com/{key}")
    if resp.status != 200:
        raise RuntimeError(f"GET {key} -> HTTP {resp.status}")
    return resp.data


# ---------------------------------------------------------------------------
# Satellite selection + S3 listing
# ---------------------------------------------------------------------------

def select_satellite(dt: date) -> str:
    """'GOES16' through the 2025-04-06 handover, 'GOES19' after."""
    return "GOES16" if dt <= GOES16_END_DATE else "GOES19"


def _list_prefix(http: urllib3.PoolManager, bucket: str, prefix: str) -> list[str]:
    """All object keys under a prefix (anonymous ListObjectsV2, paginated)."""
    keys: list[str] = []
    token: str | None = None
    while True:
        fields = {"list-type": "2", "prefix": prefix}
        if token:
            fields["continuation-token"] = token
        resp = http.request(
            "GET", f"https://{bucket}.s3.amazonaws.com/", fields=fields
        )
        if resp.status != 200:
            raise RuntimeError(f"list {prefix} -> HTTP {resp.status}")
        root = ET.fromstring(resp.data)
        keys.extend(
            el.text for el in root.iter(f"{_S3_XMLNS}Key") if el.text
        )
        truncated = root.findtext(f"{_S3_XMLNS}IsTruncated") == "true"
        if not truncated:
            return keys
        token = root.findtext(f"{_S3_XMLNS}NextContinuationToken")


def list_day_keys(http: urllib3.PoolManager, bucket: str, dt: date) -> list[str]:
    """Every GLM-L2-LCFA object key for a date (24 hourly prefixes, in parallel)."""
    doy = dt.timetuple().tm_yday
    prefixes = [f"{PRODUCT}/{dt.year}/{doy:03d}/{hour:02d}/" for hour in range(24)]
    with ThreadPoolExecutor(max_workers=len(prefixes)) as ex:
        per_hour = ex.map(lambda p: _list_prefix(http, bucket, p), prefixes)
        return [k for keys in per_hour for k in keys]


# ---------------------------------------------------------------------------
# Per-file flash extraction (from in-memory bytes)
# ---------------------------------------------------------------------------

def extract_flashes(
    data: bytes, clip: tuple[float, float, float, float] | None
) -> pd.DataFrame | None:
    """Parse one LCFA file's bytes, return its flashes (clipped), or None."""
    with _NC_LOCK:
        nc = netCDF4.Dataset("inmemory.nc", mode="r", memory=data)
        try:
            if nc.dimensions["number_of_flashes"].size == 0:
                return None
            cols: dict[str, object] = {}
            time_units: dict[str, str] = {}
            for src, out in FLASH_FIELDS.items():
                var = nc.variables[src]
                var.set_auto_mask(False)   # keep scale/offset decoding only
                cols[out] = var[:]
                if out.startswith("time_"):
                    time_units[out] = var.units
        finally:
            nc.close()

    # Offsets are float seconds since a per-file epoch ("seconds since <ts>").
    for out, units in time_units.items():
        base = pd.Timestamp(units.split("since", 1)[1].strip())
        cols[out] = base + pd.to_timedelta(cols[out], unit="s")

    df = pd.DataFrame(cols)
    if clip is not None:
        lon_min, lon_max, lat_min, lat_max = clip
        df = df[(df["lon"] >= lon_min) & (df["lon"] <= lon_max)
                & (df["lat"] >= lat_min) & (df["lat"] <= lat_max)]
    if df.empty:
        return None
    df["quality_flag"] = df["quality_flag"].astype("uint8")
    for c in ("lat", "lon", "energy", "area"):
        df[c] = df[c].astype("float32")
    return df


# ---------------------------------------------------------------------------
# Build one day -> one parquet
# ---------------------------------------------------------------------------

def _out_path(dt: date, out_dir: Path) -> Path:
    return out_dir / f"{dt.year}" / f"glm_flashes_{dt:%Y%m%d}.parquet"


def build_day(
    dt: date,
    clip_name: str = "margin",
    out_dir: Path = DEFAULT_OUT_DIR,
    workers: int = DEFAULT_WORKERS,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path | None:
    """Fetch+parse a day's GLM files in one pass, write one parquet.

    Idempotent: returns the existing path untouched unless ``overwrite``. Each
    thread GETs a file into memory and parses it immediately; only the parquet
    ever hits disk.
    """
    out_path = _out_path(dt, out_dir)
    if out_path.exists() and not overwrite:
        if verbose:
            print(f"{out_path} exists; skipping (use --overwrite).")
        return out_path

    clip = CLIP_BOXES[clip_name]
    sat = select_satellite(dt)
    bucket = SATELLITES[sat]
    http = _make_http(workers)
    keys = list_day_keys(http, bucket, dt)
    if not keys:
        if verbose:
            print(f"[no data] {dt} {sat}")
        return None
    if verbose:
        print(f"{dt} {sat}: {len(keys):,} GLM files -> streaming flashes "
              f"(clip={clip_name})")

    def _fetch_and_extract(key: str) -> pd.DataFrame | None:
        return extract_flashes(_get(http, bucket, key), clip)

    frames: list[pd.DataFrame] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch_and_extract, k) for k in keys]
        it = as_completed(futs)
        if verbose:
            it = tqdm(it, total=len(futs), desc="fetch+parse", unit="file",
                      leave=False)
        for fut in it:
            try:
                df = fut.result()
            except Exception:
                failed += 1
                continue
            if df is not None:
                frames.append(df)
    if failed:
        print(f"[warn] {dt}: {failed}/{len(keys):,} files failed after retries")

    if not frames:
        if verbose:
            print(f"{dt}: no flashes in clip box.")
        return None

    out = (pd.concat(frames, ignore_index=True)
             .sort_values("time_start")
             .reset_index(drop=True))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".parquet.tmp")    # atomic: write then rename
    out[list(FLASH_FIELDS.values())].to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    if verbose:
        size_mb = out_path.stat().st_size / 1024 ** 2
        print(f"wrote {out_path}  ({len(out):,} flashes, {size_mb:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# Date-range driver (parallel across days)
# ---------------------------------------------------------------------------

def build_range(
    start: date,
    end: date,
    clip_name: str = "margin",
    out_dir: Path = DEFAULT_OUT_DIR,
    workers: int = DEFAULT_WORKERS,
    day_workers: int = DEFAULT_DAY_WORKERS,
    overwrite: bool = False,
) -> None:
    """Build per-day flash parquets across an inclusive date range."""
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    todo = [d for d in days if overwrite or not _out_path(d, out_dir).exists()]
    print(f"{len(days)} day(s) in range | {len(days) - len(todo)} already built "
          f"| building {len(todo)}")
    if not todo:
        return

    if day_workers <= 1 or len(todo) == 1:
        for d in todo:
            build_day(d, clip_name, out_dir, workers, overwrite)
        return

    # Many days: parallelize across days with processes (each does threaded I/O).
    # Use 'spawn' — fork is unsafe here because HDF5/NetCDF inherits global
    # state across fork and deadlocks when the child reads a file.
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    ctx = mp.get_context("spawn")
    n = min(day_workers, len(todo))
    print(f"spawning {n} workers x {workers} fetch threads "
          f"(the bar advances as each day finishes)")
    ex = ProcessPoolExecutor(max_workers=day_workers, mp_context=ctx)
    futs = {ex.submit(build_day, d, clip_name, out_dir, workers, overwrite, False): d
            for d in todo}
    try:
        for fut in tqdm(as_completed(futs), total=len(futs), desc="days", unit="day"):
            fut.result()
    except KeyboardInterrupt:
        print("\n[interrupted] terminating workers...")
        for proc in list(getattr(ex, "_processes", {}).values()):
            try:
                proc.terminate()
            except Exception:
                pass
        ex.shutdown(wait=False, cancel_futures=True)
        raise SystemExit(130) from None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download GOES GLM flashes, one consolidated parquet per day."
    )
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="Build per-day flash parquet(s).")
    b.add_argument("--date", metavar="YYYY-MM-DD",
                   help="Single day to build (overrides the range).")
    b.add_argument("--start-date", default=str(DEFAULT_START_DATE),
                   metavar="YYYY-MM-DD", help="Range start (default: %(default)s).")
    b.add_argument("--end-date", default=str(DEFAULT_END_DATE),
                   metavar="YYYY-MM-DD", help="Range end (default: %(default)s).")
    b.add_argument("--clip", default="margin", choices=list(CLIP_BOXES),
                   help="Spatial clip (default: margin = CONUS + ~5 deg).")
    b.add_argument("--out", default=str(DEFAULT_OUT_DIR), metavar="DIR")
    b.add_argument("--workers", type=int, default=DEFAULT_WORKERS, metavar="N",
                   help="Fetch+parse threads per day.")
    b.add_argument("--day-workers", type=int, default=DEFAULT_DAY_WORKERS,
                   metavar="N", help="Days processed in parallel (range mode).")
    b.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    try:
        if args.date:
            build_day(date.fromisoformat(args.date), args.clip, out_dir,
                      args.workers, args.overwrite)
        else:
            build_range(date.fromisoformat(args.start_date),
                        date.fromisoformat(args.end_date), args.clip, out_dir,
                        args.workers, args.day_workers, args.overwrite)
    except KeyboardInterrupt:
        print("\n[interrupted]")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
