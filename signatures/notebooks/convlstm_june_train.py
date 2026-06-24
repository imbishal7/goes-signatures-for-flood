"""DDP trainer for the June same-day ConvLSTM (full-res, 16 bands, 24 CST frames).

Run from signatures/notebooks/:
    NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 convlstm_june_train.py --epochs 25 --batch 3
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NCCL_P2P_DISABLE", "1")          # PCIe P2P hangs on this box

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parent
while not (ROOT / "config.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
# config.py is the project's single source for the grid + constants (not a model)
from config import (  # noqa: E402
    CELL_KM, IMG_H, IMG_W, TVERSKY_ALPHA, TVERSKY_BETA, build_grid_cells, grid_transform,
)

CACHE_DIR = ROOT / "signatures/cache/convlstm_june"
CKPT_PATH = ROOT / "signatures/notebooks/convlstm_june.pt"
RESULTS_NPZ = CACHE_DIR / "ddp_results.npz"
GOES_DIR = Path("/mnt/disk4/recent-goes")
GLM_DIR = Path("/mnt/disk4/recent-glm")                  # GLM lightning flashes, 1 parquet/UTC day
N_BAND, USE_TIME = 16, True
N_CH = N_BAND + (1 if USE_TIME else 0)                   # 17
GLM_FEATS = 3                                            # per-cell daily: log count / energy / area
POOL_STRIDE = 2                                          # encoder spatial downsample
LR = 3e-4


# ---------------------------------------------------------------------------
# Model building blocks (self-contained — no dependency on notebooks/model/floodnet.py)
# Encoder -> ConvLSTM cell -> CellPool regrid -> head, + recall-favouring Tversky loss.
# ---------------------------------------------------------------------------
def conv_block(cin, cout):
    """Two 3x3 convs, each GroupNorm(8) + ReLU."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """Per-frame CNN: (B, N_CH, H, W) -> (B, 64, H/POOL_STRIDE, W/POOL_STRIDE)."""
    def __init__(self, cin=N_CH):
        super().__init__()
        self.b1 = conv_block(cin, 32)
        self.b2 = conv_block(32, 64)
        self.b3 = conv_block(64, 64)
        self.pool = nn.MaxPool2d(2)
        self.out_ch = 64

    def forward(self, x):
        x = self.pool(self.b1(x))                       # /2
        x = self.pool(self.b2(x)) if POOL_STRIDE == 4 else self.b2(x)
        return self.b3(x)


class ConvLSTMCell(nn.Module):
    """One ConvLSTM step: gates are a conv over [x, h]."""
    def __init__(self, cin, ch, k=3):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(cin + ch, 4 * ch, k, padding=k // 2)

    def forward(self, x, h, c):
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h = o.sigmoid() * c.tanh()
        return h, c


class CellPool(nn.Module):
    """Scatter-mean encoder pixels into the (R, C) cells via a fixed index."""
    def __init__(self, pix2cell_sub, grid_r, grid_c):
        super().__init__()
        n = grid_r * grid_c
        idx = torch.from_numpy(pix2cell_sub.reshape(-1).astype(np.int64)).clone()
        idx[idx < 0] = n                                # off-grid -> dump bin
        counts = torch.zeros(n + 1).scatter_add_(0, idx, torch.ones(idx.numel()))
        self.register_buffer("idx", idx)
        self.register_buffer("counts", counts.clamp_min(1.0))
        self.n, self.grid_r, self.grid_c = n, grid_r, grid_c

    def forward(self, x):                               # (B, C, h, w)
        B, C, h, w = x.shape
        out = x.new_zeros(B, C, self.n + 1)
        idx = self.idx.view(1, 1, -1).expand(B, C, -1)
        out.scatter_add_(2, idx, x.reshape(B, C, h * w))
        out = out / self.counts.view(1, 1, -1)
        return out[:, :, :self.n].reshape(B, C, self.grid_r, self.grid_c)


def pool_sub(pix2cell, stride=POOL_STRIDE):
    """Subsample the full-res pixel->cell index to the encoder's resolution."""
    s = stride
    return pix2cell[s // 2::s, s // 2::s][:IMG_H // s, :IMG_W // s]


def soft_tversky(logits, targets, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, smooth=1.0):
    """1 - soft Tversky index (recall-favouring loss on the given cells)."""
    p = torch.sigmoid(logits).float()
    t = targets.float()
    tp = (p * t).sum(); fp = (p * (1 - t)).sum(); fn = ((1 - p) * t).sum()
    return 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


class SameDayCache(Dataset):
    """One sample: raw 16-band f16 cube + per-frame CST hours + label (+ GLM per-cell map if
    use_glm). The time channel and f16->f32 cast are done on the GPU (to_input) to keep host
    RAM low (~2.9 GB/cube)."""
    def __init__(self, days, use_glm=False):
        self.days = list(days)
        self.use_glm = use_glm

    def __len__(self):
        return len(self.days)

    def __getitem__(self, i):
        d = self.days[i]
        x = np.load(CACHE_DIR / f"{d}_x.npy")                       # (T, 16, H, W) f16
        t = np.load(CACHE_DIR / f"{d}_t.npy").astype(np.float32)    # (T,) CST hours
        y = np.load(CACHE_DIR / f"{d}_y.npy").astype(np.float32)    # (R, C)
        glm = (np.load(CACHE_DIR / f"{d}_glm.npy") if self.use_glm  # (GLM_FEATS, R, C)
               else np.zeros(1, np.float32))                        # placeholder, ignored
        return (torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(y),
                torch.from_numpy(glm))


def to_input(xb, hb, dev):
    """f16 cube -> f32 + CST-hour channel, assembled on the GPU -> (B, T, 17, H, W)."""
    xb = xb.to(dev, non_blocking=True).float()                      # (B,T,16,H,W) on GPU
    B, T, _, H, W = xb.shape
    if USE_TIME:
        tch = (hb.to(dev).float() / 24.0).view(B, T, 1, 1, 1).expand(B, T, 1, H, W)
        xb = torch.cat([xb, tch], dim=2)                            # (B,T,17,H,W)
    return xb


class ConvLSTMNetCkpt(nn.Module):
    """Encoder -> ConvLSTM -> CellPool -> conv head, gradient-checkpointed so all 24 full-res
    frames fit in VRAM. The per-cell logit comes from the pooled GOES features; if glm_ch>0,
    the GLM per-cell map (count/energy/area) is concatenated at the cell grid before the head."""
    def __init__(self, sub, gr, gc, cin=N_CH, ch=64, glm_ch=0):
        super().__init__()
        self.encoder = Encoder(cin)
        self.cell = ConvLSTMCell(self.encoder.out_ch, ch)
        self.ch = ch
        self.glm_ch = glm_ch
        self.pool = CellPool(sub, gr, gc)
        self.head = nn.Sequential(conv_block(ch + glm_ch, 64), nn.Conv2d(64, 1, 1))

    def forward(self, x, glm=None):                                  # x (B, T, C, H, W)
        B, T = x.shape[:2]
        feats = [checkpoint(self.encoder, x[:, t], use_reentrant=False) for t in range(T)]
        h = x.new_zeros(B, self.ch, *feats[0].shape[2:])
        c = torch.zeros_like(h)
        for t in range(T):
            h, c = checkpoint(self.cell, feats[t], h, c, use_reentrant=False)
        cell = self.pool(h)                                          # (B, ch, R, C)
        if self.glm_ch:
            cell = torch.cat([cell, glm], dim=1)                     # + GLM per-cell map
        return self.head(cell).squeeze(1)                            # (B, R, C)


def build_sub():
    """50 km grid + full-res pixel->cell index (cached; rebuilt from a GOES-19 frame if absent)."""
    import pyproj
    import netCDF4
    cells, gr, gc, land = build_grid_cells()
    cache = CACHE_DIR / f"pix2cell_{CELL_KM}km.npy"
    if cache.exists():
        p2c = np.load(cache)
    else:
        gx0, gy0, gstep, *_ = grid_transform()
        ref = sorted(GOES_DIR.glob("*/2026/*/*/*.nc"))[0]
        with netCDF4.Dataset(ref) as nc:
            proj = nc["goes_imager_projection"]
            geos = pyproj.CRS.from_cf({k: proj.getncattr(k) for k in proj.ncattrs()})
            h = float(proj.perspective_point_height)
            xc, yc = nc["x"][:].astype(np.float64) * h, nc["y"][:].astype(np.float64) * h
        xx, yy = np.meshgrid(xc, yc)
        ax, ay = pyproj.Transformer.from_crs(geos, 5070, always_xy=True).transform(
            xx.ravel(), yy.ravel())
        ax, ay = np.asarray(ax).reshape(xx.shape), np.asarray(ay).reshape(xx.shape)
        col = np.floor((ax - gx0) / gstep)
        row = (gr - 1) - np.floor((ay - gy0) / gstep)
        ok = (np.isfinite(ax) & np.isfinite(ay)
              & (col >= 0) & (col < gc) & (row >= 0) & (row < gr))
        p2c = np.full(ax.shape, -1, np.int32)
        p2c[ok] = (row[ok] * gc + col[ok]).astype(np.int32)
        np.save(cache, p2c)
    return pool_sub(p2c, stride=2), gr, gc, land


def splits():
    m = pd.read_parquet(CACHE_DIR / "manifest.parquet")
    g = lambda s: [d.strftime("%Y%m%d") for d in m.loc[m.split == s, "cst_day"]]  # noqa: E731
    return g("train"), g("val"), g("test")


def build_glm_cache(days):
    """Bin GLM flashes to the 50 km grid per CST day -> {day}_glm.npy (GLM_FEATS, R, C).
    Features are log1p of per-cell daily flash count, total energy, total area. CST = UTC-6,
    matching the GOES/label binning (a CST day spans two UTC parquet files)."""
    import pyproj
    cells, gr, gc, land = build_grid_cells()
    x0, y0, step, *_ = grid_transform()
    tf = pyproj.Transformer.from_crs(4326, 5070, always_xy=True)
    cst = pd.Timedelta(hours=6)
    n = gr * gc
    for ds in days:
        out = CACHE_DIR / f"{ds}_glm.npy"
        if out.exists():
            continue
        d0 = pd.Timestamp(ds)
        frames = []
        for k in (0, 1):                                   # UTC day d and d+1 cover CST day d
            f = GLM_DIR / "2026" / f"glm_flashes_{(d0 + pd.Timedelta(days=k)):%Y%m%d}.parquet"
            if f.exists():
                frames.append(pd.read_parquet(f, columns=["time_start", "lat", "lon",
                                                           "energy", "area"]))
        feats = np.zeros((GLM_FEATS, gr, gc), np.float32)
        if frames:
            df = pd.concat(frames, ignore_index=True)
            loc = df["time_start"] - cst
            df = df[(loc >= d0) & (loc < d0 + pd.Timedelta(days=1))]
            if len(df):
                ax, ay = tf.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
                col = np.floor((np.asarray(ax) - x0) / step).astype(np.int64)
                row = (gr - 1) - np.floor((np.asarray(ay) - y0) / step).astype(np.int64)
                ok = (col >= 0) & (col < gc) & (row >= 0) & (row < gr)
                flat = row[ok] * gc + col[ok]
                cnt = np.bincount(flat, minlength=n).astype(np.float64)
                eng = np.bincount(flat, weights=df["energy"].to_numpy()[ok].astype(np.float64), minlength=n)
                ar = np.bincount(flat, weights=df["area"].to_numpy()[ok].astype(np.float64), minlength=n)
                feats[0] = np.log1p(cnt).reshape(gr, gc)
                feats[1] = np.log1p(eng * 1e15).reshape(gr, gc)
                feats[2] = np.log1p(ar / 1e8).reshape(gr, gc)
        np.save(out, feats)


@torch.no_grad()
def evalset(net, days, dev, use_glm=False):
    net.eval(); P, Y = [], []
    for xb, hb, yb, gb in DataLoader(SameDayCache(days, use_glm), 1, num_workers=0, pin_memory=False):
        glm = gb.to(dev).float() if use_glm else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(net(to_input(xb, hb, dev), glm)).float().cpu()
        P.append(p.numpy()); Y.append(yb.numpy())
    return np.concatenate(P), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=1)            # per GPU (~3 GB/sample)
    ap.add_argument("--workers", type=int, default=0)          # 0 = no prefetch (RAM-safe)
    ap.add_argument("--glm", action="store_true",
                    help="add GLM lightning (per-cell daily count/energy/area) at the cell grid")
    ap.add_argument("--tag", default="",
                    help="suffix for the output files (e.g. 'glm' -> ddp_results_glm.npz)")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    results_npz = CACHE_DIR / f"ddp_results{suffix}.npz"
    ckpt_path = ROOT / "signatures/notebooks" / f"convlstm_june{suffix}.pt"

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local); dev = torch.device("cuda", local)
    torch.set_float32_matmul_precision("high")

    sub, gr, gc, land = build_sub()
    tr_days, va_days, te_days = splits()
    mask = torch.from_numpy(land).to(dev)

    glm_ch = GLM_FEATS if args.glm else 0
    if args.glm and rank == 0:                             # build the GLM cache once (rank 0)
        build_glm_cache(tr_days + va_days + te_days)
    dist.barrier()

    net = ConvLSTMNetCkpt(sub, gr, gc, cin=N_CH, glm_ch=glm_ch).to(dev)
    model = DDP(net, device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    tr_ds = SameDayCache(tr_days, args.glm)
    smp = DistributedSampler(tr_ds, shuffle=True)
    tr_dl = DataLoader(tr_ds, args.batch, sampler=smp, num_workers=args.workers,
                       pin_memory=False)

    if rank == 0:
        print(f"[ddp] world={world} params {sum(p.numel() for p in net.parameters()):,} | "
              f"inputs={'GOES+GLM' if args.glm else 'GOES (images-only)'} | "
              f"train {len(tr_days)} val {len(va_days)} test {len(te_days)} | "
              f"batch/gpu {args.batch} | loss Tversky a={TVERSKY_ALPHA} b={TVERSKY_BETA}",
              flush=True)

    hist, t0 = [], time.perf_counter()
    for ep in range(1, args.epochs + 1):
        smp.set_epoch(ep); model.train(); tot = torch.zeros((), device=dev)
        for xb, hb, yb, gb in tr_dl:
            yb = yb.to(dev)
            glm = gb.to(dev).float() if args.glm else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = soft_tversky(model(to_input(xb, hb, dev), glm)[:, mask], yb[:, mask],
                                    TVERSKY_ALPHA, TVERSKY_BETA)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.detach()
        dist.all_reduce(tot); tr_loss = (tot / (len(tr_dl) * world)).item()
        if rank == 0:
            P, Y = evalset(net, te_days, dev, args.glm)   # no val split -> monitor on test
            tap = average_precision_score(Y[:, land].ravel(), P[:, land].ravel())
            hist.append((ep, tr_loss, tap))
            print(f"[ddp] epoch {ep:2d}/{args.epochs}  train {tr_loss:.4f}  "
                  f"TEST PR-AUC {tap:.4f}  [{(time.perf_counter()-t0)/60:.1f} min]  "
                  f"peak {torch.cuda.max_memory_allocated(local)/1e9:.0f}GB", flush=True)
        dist.barrier()

    if rank == 0:
        gr_, gc_ = land.shape
        if va_days:
            vp, vy = evalset(net, va_days, dev, args.glm)
        else:
            vp = np.zeros((0, gr_, gc_), np.float32); vy = np.zeros((0, gr_, gc_), np.uint8)
        np.savez(results_npz, hist=np.array(hist, np.float32),
                 probs=P.astype(np.float32), trues=Y.astype(np.uint8),
                 te_days=np.array(te_days),
                 val_probs=vp.astype(np.float32), val_trues=vy.astype(np.uint8),
                 va_days=np.array(va_days))
        torch.save(net.state_dict(), ckpt_path)
        ap_te = average_precision_score(Y[:, land].ravel(), P[:, land].ravel())
        print(f"[ddp] TEST PR-AUC {ap_te:.4f}  -> saved {ckpt_path.name} + {results_npz.name}",
              flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
