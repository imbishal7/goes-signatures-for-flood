"""
GOES ABI-L2-MCMIPC downloader for GOES-16 and GOES-19.

Band reference:
  Geocolor    : bands 3, 2, 1
  Cloud props : band 6
  All 16 bands are included in each ABI-L2-MCMIPC NetCDF file.

S3 path layout (identical for both satellites):
  s3://noaa-goes{16|19}/ABI-L2-MCMIPC/{year}/{doy:03d}/{hour:02d}/
    OR_ABI-L2-MCMIPC-M6_G{16|19}_s{YYYYDDDHHMMSSf}_e{...}_c{...}.nc

Satellite cutover:
  2020-01-01 → 2025-04-06  GOES-16  (noaa-goes16)
  2025-04-07 → present      GOES-19  (noaa-goes19)
"""

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SATELLITES: dict[str, str] = {
    "GOES16": "noaa-goes16",
    "GOES19": "noaa-goes19",
}

PRODUCT = "ABI-L2-MCMIPC"

GOES16_END_DATE = date(2025, 4, 6)
GOES19_START_DATE = date(2025, 4, 7)

# 8 images per CST day at 3-hour gaps (00, 03, ..., 21 CST). GOES files are
# UTC-named and CST = UTC-6 (DST ignored); since 6 h is a multiple of the 3 h
# gap, the UTC target hours are the same 8 values 00..21.
DEFAULT_TARGET_HOURS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
TYPICAL_FILE_SIZE_MB = 60.0   # verified: 57–61 MB per file

# default full span; override per run with --start / --end
DEFAULT_START = date(2015, 1, 1)
DEFAULT_END = date(2018, 12, 30)

# NOTE: downloads default to /mnt/disk4 (overflow). The model pipeline reads GOES
# from /mnt/disk1/goes-data (config.DATA_DIR) — pass --data-dir to match, or
# move/symlink, so the notebooks find what you pull here.
DEFAULT_DATA_DIR = Path("/mnt/disk4/recent-goes")
DEFAULT_WORKERS = 25           # saturates 100 MB/s with ~60 MB files


# ---------------------------------------------------------------------------
# S3 client
# ---------------------------------------------------------------------------

_thread_local = threading.local()

def make_s3_client():
    """Return an anonymous boto3 S3 client for NOAA's public GOES buckets."""
    return boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )

def _get_thread_s3_client():
    """Return a per-thread S3 client, creating it on first use in each thread."""
    if not hasattr(_thread_local, "s3"):
        _thread_local.s3 = make_s3_client()
    return _thread_local.s3


# ---------------------------------------------------------------------------
# Satellite selection
# ---------------------------------------------------------------------------

def select_satellite(dt: date) -> str:
    """Return 'GOES16' for dates up to the handover, 'GOES19' after."""
    return "GOES16" if dt <= GOES16_END_DATE else "GOES19"


# ---------------------------------------------------------------------------
# S3 file listing
# ---------------------------------------------------------------------------

def list_files(s3_client, satellite_key: str, dt: date, hour: int) -> list[dict]:
    """
    List ABI-L2-MCMIPC objects in S3 for a given date and UTC hour.

    Returns a list of dicts with keys 'key', 'size', 'last_modified'.
    Returns [] if the prefix is empty or any S3 error occurs.
    """
    bucket = SATELLITES[satellite_key]
    doy = dt.timetuple().tm_yday
    prefix = f"{PRODUCT}/{dt.year}/{doy:03d}/{hour:02d}/"

    results = []
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                results.append({
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"],
                })
    except ClientError:
        return []
    return results


# ---------------------------------------------------------------------------
# File selection within an hour
# ---------------------------------------------------------------------------

def select_closest_file(files: list[dict], target_minute: int = 0) -> dict | None:
    """
    Return the file whose scan start minute is closest to target_minute.

    Filename encodes start time as s{YYYYDDDHHMMSSf}; parses the minute field.
    Returns None if files is empty.
    """
    if not files:
        return None

    def scan_minute(f: dict) -> int:
        name = os.path.basename(f["key"])
        # token like s20200011801174 → YYYY[1:5] DDD[5:8] HH[8:10] MM[10:12]
        token = [p for p in name.split("_") if p.startswith("s")][0]
        return int(token[10:12])

    return min(files, key=lambda f: abs(scan_minute(f) - target_minute))


# ---------------------------------------------------------------------------
# Local path construction
# ---------------------------------------------------------------------------

def build_local_path(
    base_dir: Path, satellite_key: str, dt: date, filename: str
) -> Path:
    """Return data/goes/GOES16/YYYY/MM/DD/<filename>."""
    return (
        base_dir
        / satellite_key
        / f"{dt.year}"
        / f"{dt.month:02d}"
        / f"{dt.day:02d}"
        / filename
    )


# ---------------------------------------------------------------------------
# File download
# ---------------------------------------------------------------------------

def download_file(
    s3_client, bucket: str, s3_key: str, local_path: Path, show_progress: bool = True
) -> None:
    """Download one S3 object to local_path, optionally with a tqdm progress bar."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if show_progress:
            file_size = s3_client.head_object(
                Bucket=bucket, Key=s3_key
            )["ContentLength"]
            with tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=local_path.name,
                leave=False,
            ) as bar:
                s3_client.download_file(
                    Bucket=bucket,
                    Key=s3_key,
                    Filename=str(local_path),
                    Callback=lambda n: bar.update(n),
                )
        else:
            s3_client.download_file(Bucket=bucket, Key=s3_key, Filename=str(local_path))
    except Exception:
        if local_path.exists():
            local_path.unlink()
        raise


# ---------------------------------------------------------------------------
# Storage estimator
# ---------------------------------------------------------------------------

def estimate_storage(
    start_date: date = DEFAULT_START,
    end_date: date = DEFAULT_END,
    images_per_day: int = 1,
    file_size_mb: float = TYPICAL_FILE_SIZE_MB,
) -> dict:
    """Print and return a storage breakdown for the requested date range."""
    goes16_days = 0
    goes19_days = 0
    current = start_date
    while current <= end_date:
        if select_satellite(current) == "GOES16":
            goes16_days += 1
        else:
            goes19_days += 1
        current += timedelta(days=1)

    total_days = goes16_days + goes19_days

    def gb(days: int) -> float:
        return days * images_per_day * file_size_mb / 1024

    print("\nGOES Storage Estimate")
    print(f"Date range  : {start_date} to {end_date} ({total_days} days)")
    print(f"File size   : ~{file_size_mb:.0f} MB per ABI-L2-MCMIPC file")
    print(f"Images/day  : {images_per_day}")
    print()
    print(f"{'':20s} {'GOES-16':>10} {'GOES-19':>10} {'Total':>10}")
    print(
        f"{'Days with data':20s} {goes16_days:>10,}"
        f" {goes19_days:>10,} {total_days:>10,}"
    )
    print(
        f"{'Files':20s} {goes16_days * images_per_day:>10,}"
        f" {goes19_days * images_per_day:>10,}"
        f" {total_days * images_per_day:>10,}"
    )
    print(
        f"{'Storage':20s} {gb(goes16_days):>9.1f}G"
        f" {gb(goes19_days):>9.1f}G {gb(total_days):>9.1f}G"
    )
    print()

    if images_per_day == 1:
        n = len(DEFAULT_TARGET_HOURS)
        print(f"--- {n} images/day estimate ---")
        print(
            f"{'Files':20s} {goes16_days * n:>10,}"
            f" {goes19_days * n:>10,} {total_days * n:>10,}"
        )
        print(
            f"{'Storage':20s} {gb(goes16_days) * n:>9.1f}G"
            f" {gb(goes19_days) * n:>9.1f}G"
            f" {gb(total_days) * n:>9.1f}G"
        )
        print()

    return {
        "total_days": total_days,
        "goes16_days": goes16_days,
        "goes19_days": goes19_days,
        "total_files": total_days * images_per_day,
        "total_gb": gb(total_days),
        "per_satellite": {
            "GOES16": {
                "days": goes16_days,
                "files": goes16_days * images_per_day,
                "gb": gb(goes16_days),
            },
            "GOES19": {
                "days": goes19_days,
                "files": goes19_days * images_per_day,
                "gb": gb(goes19_days),
            },
        },
    }


# ---------------------------------------------------------------------------
# Single-task worker (runs in each thread)
# ---------------------------------------------------------------------------

def _download_task(
    dt: date,
    hour: int,
    base_dir: Path,
    dry_run: bool,
    verbose: bool,
) -> tuple[str, int]:
    """
    Resolve and download one (date, hour) slot.

    Returns (status, size_bytes). status is one of 'downloaded', 'dry_run',
    'skipped_existing', 'skipped_no_data', or 'error:<msg>'. size_bytes is the
    chosen file's real S3 size for 'downloaded'/'dry_run', else 0.
    """
    satellite_key = select_satellite(dt)
    bucket = SATELLITES[satellite_key]
    sat_code = "G16" if satellite_key == "GOES16" else "G19"
    doy = dt.timetuple().tm_yday
    local_dir = build_local_path(base_dir, satellite_key, dt, "").parent

    # Resume: skip if a matching file for this date/hour already exists
    existing = list(local_dir.glob(
        f"OR_ABI-L2-MCMIPC-M6_{sat_code}_s{dt.year}{doy:03d}{hour:02d}*.nc"
    ))
    if existing:
        return "skipped_existing", 0

    s3 = _get_thread_s3_client()
    files = list_files(s3, satellite_key, dt, hour)
    chosen = select_closest_file(files)

    if chosen is None:
        if verbose:
            tqdm.write(f"  [no data] {dt} {satellite_key} hour={hour:02d}")
        return "skipped_no_data", 0

    filename = os.path.basename(chosen["key"])
    local_path = build_local_path(base_dir, satellite_key, dt, filename)
    size_mb = chosen["size"] / 1024 / 1024

    if dry_run:
        tqdm.write(
            f"[DRY-RUN] s3://{bucket}/{chosen['key']}  ({size_mb:.1f} MB)"
            f"  → {local_path}"
        )
        return "dry_run", chosen["size"]

    try:
        download_file(s3, bucket, chosen["key"], local_path, show_progress=False)
        return "downloaded", chosen["size"]
    except Exception as exc:
        tqdm.write(f"  [error] {dt} {satellite_key} hour={hour:02d}: {exc}")
        return f"error:{exc}", 0


# ---------------------------------------------------------------------------
# Main download loop
# ---------------------------------------------------------------------------

def download_date_range(
    start_date: date = DEFAULT_START,
    end_date: date = DEFAULT_END,
    base_dir: Path = DEFAULT_DATA_DIR,
    target_hours: list[int] | None = None,
    workers: int = DEFAULT_WORKERS,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Download one ABI-L2-MCMIPC file per day (per target hour) across a date range.

    Uses GOES-16 through 2025-04-06, GOES-19 from 2025-04-07 onward.
    Skips already-downloaded files (idempotent re-runs).
    Defaults to 8 images per CST day (3-hour gaps); pass target_hours=[18] for one.
    Workers run in threads — boto3 clients are thread-safe.
    With dry_run=True, nothing is fetched; it lists every file and reports the
    exact total download size from real S3 object sizes.
    """
    if target_hours is None:
        target_hours = DEFAULT_TARGET_HOURS

    # Build the full task list upfront so tqdm can show accurate totals
    tasks: list[tuple[date, int]] = []
    current = start_date
    while current <= end_date:
        for hour in target_hours:
            tasks.append((current, hour))
        current += timedelta(days=1)

    stats: dict[str, int] = {
        "downloaded": 0, "skipped_existing": 0,
        "skipped_no_data": 0, "errors": 0, "dry_run": 0,
    }
    total_bytes = 0
    stats_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_task, dt, hour, base_dir, dry_run, verbose
            ): (dt, hour)
            for dt, hour in tasks
        }
        with tqdm(total=len(futures), desc="Files", unit="file") as bar:
            for future in as_completed(futures):
                result, size = future.result()
                with stats_lock:
                    if result == "downloaded":
                        stats["downloaded"] += 1
                        total_bytes += size
                    elif result == "dry_run":
                        stats["dry_run"] += 1
                        total_bytes += size
                    elif result == "skipped_existing":
                        stats["skipped_existing"] += 1
                    elif result == "skipped_no_data":
                        stats["skipped_no_data"] += 1
                    elif result.startswith("error:"):
                        stats["errors"] += 1
                bar.update(1)

    stats["total_bytes"] = total_bytes

    if verbose:
        if dry_run:
            gb = total_bytes / 1024 ** 3
            print(
                f"\n[DRY-RUN] would download {stats['dry_run']:,} files"
                f" = {total_bytes:,} bytes ({gb:.2f} GB, exact)."
                f"  skipped_existing={stats['skipped_existing']}"
                f"  skipped_no_data={stats['skipped_no_data']}"
            )
        else:
            print(
                f"\nDone. downloaded={stats['downloaded']}  "
                f"skipped_existing={stats['skipped_existing']}  "
                f"skipped_no_data={stats['skipped_no_data']}  "
                f"errors={stats['errors']}"
            )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GOES ABI-L2-MCMIPC imagery from AWS S3."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- download subcommand -------------------------------------------------
    dl = sub.add_parser("download", help="Download files for a date range.")
    dl.add_argument(
        "--start-date", default=str(DEFAULT_START), metavar="YYYY-MM-DD",
        help=f"First date to download (default: {DEFAULT_START})",
    )
    dl.add_argument(
        "--end-date", default=str(DEFAULT_END), metavar="YYYY-MM-DD",
        help=f"Last date to download (default: {DEFAULT_END})",
    )
    dl.add_argument(
        "--data-dir", default=str(DEFAULT_DATA_DIR), metavar="PATH",
        help=f"Root directory for downloads (default: {DEFAULT_DATA_DIR})",
    )
    dl.add_argument(
        "--hour", type=int, nargs="+", default=DEFAULT_TARGET_HOURS, metavar="H",
        help=(
            f"UTC hour(s) to target (default: {DEFAULT_TARGET_HOURS}, "
            "8 images per CST day at 3-hour gaps). Pass one for 1/day, e.g. --hour 18"
        ),
    )
    dl.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, metavar="N",
        help=f"Parallel download threads (default: {DEFAULT_WORKERS})",
    )
    dl.add_argument(
        "--dry-run", action="store_true", help="List files without downloading."
    )
    dl.add_argument("--quiet", action="store_true", help="Suppress per-file messages.")

    # -- estimate subcommand -------------------------------------------------
    est = sub.add_parser("estimate", help="Print storage estimate and exit.")
    est.add_argument("--start-date", default=str(DEFAULT_START), metavar="YYYY-MM-DD")
    est.add_argument("--end-date", default=str(DEFAULT_END), metavar="YYYY-MM-DD")
    est.add_argument(
        "--images-per-day", type=int, default=1, metavar="N",
        help="Images per day (default: 1)",
    )
    est.add_argument(
        "--file-size-mb", type=float, default=TYPICAL_FILE_SIZE_MB, metavar="MB",
        help=f"Assumed file size in MB (default: {TYPICAL_FILE_SIZE_MB})",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "estimate":
        estimate_storage(
            start_date=date.fromisoformat(args.start_date),
            end_date=date.fromisoformat(args.end_date),
            images_per_day=args.images_per_day,
            file_size_mb=args.file_size_mb,
        )

    elif args.command == "download":
        download_date_range(
            start_date=date.fromisoformat(args.start_date),
            end_date=date.fromisoformat(args.end_date),
            base_dir=Path(args.data_dir),
            target_hours=args.hour,
            workers=args.workers,
            dry_run=args.dry_run,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
