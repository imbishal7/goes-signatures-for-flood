# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project status

**In progress / early research.** The research question is **not finalized yet**. The
work centers on using **GOES satellite imagery to detect spectral signatures that
precede or indicate extreme flooding across the contiguous United States (CONUS)**.
Treat the exact prediction target, modeling approach, and evaluation as still open —
flag assumptions rather than locking them in, and expect direction to shift as the
data is explored.

## Hardware — use it well

This is a single high-end workstation. **For every task in this project, prefer
solutions that actually exploit this hardware** (parallel/multiprocessing over the
many cores, GPU acceleration for any modeling/array work, generous in-memory
processing, large I/O parallelism). Do not write single-threaded, CPU-only,
memory-shy code by default.

| Resource | Spec |
|---|---|
| CPU | AMD Ryzen Threadripper PRO 9985WX — **64 physical cores / 128 threads** |
| RAM | **128 GB** |
| GPU | **2× NVIDIA RTX PRO 6000 Blackwell** (Max-Q Workstation Edition), **~96 GB VRAM each** (~192 GB total) |
| Storage | **~10 TB across separate drives** (see below) |

> Note: the CPU is 64 *cores* with SMT giving 128 *threads* (the OS reports 128
> logical CPUs). Both GPUs report ~96 GB VRAM, so multi-GPU and large-batch /
> large-model workloads are very feasible.

Practical defaults:
- Parallelize across cores (e.g. `ProcessPoolExecutor`, `joblib`, Dask, or library
  thread/worker counts) — the download script already defaults to many workers.
- Use the GPUs for model training/inference and heavy array math — the stack is
  installed (PyTorch, CuPy, RAPIDS; see **GPU / ML stack** below). Both cards are
  available for data-parallel (DDP) training or running two jobs/models at once.
- Memory is plentiful — batch large reads rather than trickling them.

## Storage layout — write to `/mnt/disk1` only (overflow to `/mnt/disk4`)

Drives are mounted as `/mnt/disk1` … `/mnt/disk4` (each ~1.8 TB usable) plus the ~1 TB
root NVMe.

- **All of this project's data and generated artifacts go on `/mnt/disk1`.** GOES imagery
  already lives at `/mnt/disk1/goes-data/` (~919 GB / ~15.6k NetCDF files); put derived
  datasets, caches, features, model checkpoints, etc. under `/mnt/disk1` too.
- **`/mnt/disk4` is the ONLY overflow** — use it only when `/mnt/disk1` cannot hold any
  more. Check free space first (`df -h /mnt/disk1`) before assuming disk1 has room.
- **Do NOT write to `/mnt/disk2` or `/mnt/disk3`** — they hold other projects' data
  (e.g. FloodSimBench, hydrofabric) and are off-limits.
- Keep large/generated artifacts **off the git tree** — `data/raw/` and the repo only
  hold the small ground-truth parquet and code; large files are gitignored.

## Data

### Ground truth — flood events
`data/raw/groundsource_2026.parquet` (~637 MB). ~2.65M rows, columns:
`uuid`, `area_km2`, `geometry` (WKB-encoded polygons/multipolygons), `start_date`,
`end_date` (dates stored as strings). Source: Zenodo record 18647054.

### GOES imagery
NOAA ABI-L2-MCMIPC (multi-band cloud & moisture, CONUS sector — all 16 ABI bands per
~60 MB NetCDF file), pulled from public AWS S3 (`noaa-goes16` / `noaa-goes19`, no
credentials). Satellite cutover: **GOES-16** through 2025-04-06, **GOES-19** from
2025-04-07 on. Default pull is **6 daytime images/day** (16–21 UTC) →
`/mnt/disk1/goes-data/GOES{16|19}/YYYY/MM/DD/`. Downloads are resumable/idempotent.

## Repo structure & key files

```
src/goes_data.py          # GOES S3 downloader + storage estimator (CLI + importable)
notebooks/explore.ipynb   # EDA on ground-truth flood events (spatial/temporal/area)
notebooks/preview_goes.ipynb  # interactive viewer for downloaded GOES imagery (folium overlay, RGB, crop)
data/raw/                 # ground-truth parquet only (large data gitignored)
pyproject.toml / uv.lock  # deps, managed by uv
README.md                 # user-facing setup & download docs
```

`src/goes_data.py` is the most important module so far — it handles satellite
selection by date, per-thread S3 clients, closest-to-target-minute file selection,
resumable parallel downloads, and an `estimate` vs `--dry-run` (exact size) split.

## Environment & commands

Python ≥ 3.11, managed with **uv**. Run things via `uv run ...` (or the project
`.venv`).

```bash
uv sync                                            # install deps
uv run jupyter lab                                 # notebooks
uv run python src/goes_data.py estimate            # rough storage projection
uv run python src/goes_data.py download --dry-run  # exact size, no fetch
uv run python src/goes_data.py download            # 6 daytime imgs/day -> /mnt/disk1
```

Lint: **ruff** (line length 88, rules `E`,`F`,`I`). Tests: pytest (none yet).

## GPU / ML stack

Installed and **verified to compute on both Blackwell (sm_120) cards**. Because these
GPUs are sm_120, anything CUDA must use **CUDA 12.8+ wheels** — older builds won't run.

**Locked (in `pyproject.toml`, installed by `uv sync`):**
- `torch` 2.11 / `torchvision` (cu128) — pulled from the `pytorch-cu128` index
  (configured under `[tool.uv.sources]`). bf16 is supported — prefer it for training.
- `lightning` — multi-GPU training; use the **DDP** strategy to span both cards
  (`Trainer(accelerator="gpu", devices=2, strategy="ddp", precision="bf16-mixed")`).
  NCCL is available.
- `cupy-cuda12x` — drop-in GPU NumPy.
- `timm` (vision backbones), `einops`, `xbatcher` (xarray→ML batches for the GOES
  NetCDFs), `tensorboard`.

**RAPIDS overlay (NOT locked — installed via pip):** `cudf` + `cuspatial` + `cuml`
(all `25.4.*`) plus `dask-cuda`. cuSpatial is the relevant piece for the 2.65M flood
polygons (GPU point-in-polygon / spatial joins against GOES pixel grids); cuML gives
GPU KMeans/PCA/etc. Install / restore with:

```bash
.venv/bin/python -m pip install --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.4.*" "cuspatial-cu12==25.4.*" "cuml-cu12==25.4.*"
```

Gotchas:
- **Pinned to RAPIDS 25.4** because cuSpatial's newest cp312 wheel is 25.4.0; cuDF/cuML
  must match it (newer cuDF/cuML exist but break the trio).
- RAPIDS pins **numpy down to 2.0.x** (and pandas 2.2 / pyarrow 19). torch & CuPy work
  fine at numpy 2.0.2 — verified — so this is harmless, just don't "upgrade numpy".
- A plain **`uv sync` will drop the RAPIDS overlay** and bump numpy back. Re-run the
  pip line above afterward, or use `uv sync --inexact` to leave the overlay in place.

Quick sanity check: `import torch; torch.cuda.device_count()` → `2`; `import cudf, cuspatial, cuml`.

## Conventions

- Match the existing style in `src/goes_data.py`: typed signatures, clear docstrings,
  section banner comments, stdlib-first.
- Default new heavy compute to multi-core / GPU paths (see Hardware).
- Don't commit large data; write derived artifacts under `/mnt/disk1` (overflow to
  `/mnt/disk4` only), never `/mnt/disk2`–`disk3` or the repo.
