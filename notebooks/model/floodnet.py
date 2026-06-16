"""FloodConvLSTM model, dataset, loss, and the pixel->cell index.

Single source of truth shared by ``train_ddp.py`` (DDP training) and notebook 02
(metrics + predictions), so a checkpoint trained by the script loads identically
in the notebook. The encoder downsample factor lives here (``POOL_STRIDE``), not
in config — it is a model knob.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# repo-root config (single source of truth for paths/constants)
ROOT = Path(__file__).resolve().parent
while not (ROOT / "config.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from config import (  # noqa: E402
    CACHE_DIR,
    CELL_KM,
    DATA_DIR,
    IMG_H,
    IMG_W,
    N_CH,
    TVERSKY_ALPHA,
    TVERSKY_BETA,
    USE_TIME,
    build_grid_cells,
    grid_transform,
)

POOL_STRIDE = 4                       # encoder spatial downsample (model knob)


# ---------------------------------------------------------------------------
# Pixel -> cell index (geographic regrid lookup)
# ---------------------------------------------------------------------------
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
    np.save(cache, p2c)
    return p2c, grid_r, grid_c, land_mask


def pool_sub(pix2cell, stride=POOL_STRIDE):
    """Subsample the full-res index to the encoder's resolution (block centres)."""
    s = stride
    return pix2cell[s // 2::s, s // 2::s][:IMG_H // s, :IMG_W // s]


# ---------------------------------------------------------------------------
# Model: encoder (per frame) -> ConvLSTM (time) -> CellPool (grid) -> head
# ---------------------------------------------------------------------------
def conv_block(cin, cout):
    """Two 3x3 convs, each GroupNorm(8) + ReLU."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """Per-frame CNN: (B, N_CH, 1500, 2500) -> (B, 64, 375, 625), downsampled /4."""
    def __init__(self, cin=N_CH):
        super().__init__()
        self.b1 = conv_block(cin, 32)
        self.b2 = conv_block(32, 64)
        self.b3 = conv_block(64, 64)
        self.pool = nn.MaxPool2d(2)
        self.out_ch, self.stride = 64, POOL_STRIDE

    def forward(self, x):
        x = self.pool(self.b1(x))        # /2
        x = self.pool(self.b2(x))        # /4
        return self.b3(x)                # refine


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


class ConvLSTM(nn.Module):
    """Walk the T frames in order; return the final hidden feature map."""
    def __init__(self, cin, ch=64, k=3):
        super().__init__()
        self.cell = ConvLSTMCell(cin, ch, k)
        self.ch = ch

    def forward(self, seq):                  # (B, T, cin, H, W)
        B, T, _, H, W = seq.shape
        h = seq.new_zeros(B, self.ch, H, W)
        c = seq.new_zeros(B, self.ch, H, W)
        for t in range(T):
            h, c = self.cell(seq[:, t], h, c)
        return h


class CellPool(nn.Module):
    """Scatter-mean encoder pixels into the (R, C) cells via a fixed index."""
    def __init__(self, pix2cell_sub, grid_r, grid_c):
        super().__init__()
        n = grid_r * grid_c
        idx = torch.from_numpy(pix2cell_sub.reshape(-1).astype(np.int64)).clone()
        idx[idx < 0] = n                                  # off-grid -> dump bin
        counts = torch.zeros(n + 1).scatter_add_(0, idx, torch.ones(idx.numel()))
        self.register_buffer("idx", idx)
        self.register_buffer("counts", counts.clamp_min(1.0))
        self.n, self.grid_r, self.grid_c = n, grid_r, grid_c

    def forward(self, x):                                 # (B, C, h, w)
        B, C, h, w = x.shape
        out = x.new_zeros(B, C, self.n + 1)
        idx = self.idx.view(1, 1, -1).expand(B, C, -1)
        out.scatter_add_(2, idx, x.reshape(B, C, h * w))
        out = out / self.counts.view(1, 1, -1)            # per-cell mean
        return out[:, :, :self.n].reshape(B, C, self.grid_r, self.grid_c)


class FloodConvLSTM(nn.Module):
    """encoder (per frame) -> ConvLSTM (time) -> CellPool (grid) -> head -> logits."""
    def __init__(self, pix2cell_sub, grid_r, grid_c, cin=N_CH):
        super().__init__()
        self.encoder = Encoder(cin)
        self.convlstm = ConvLSTM(self.encoder.out_ch, ch=64)
        self.pool = CellPool(pix2cell_sub, grid_r, grid_c)
        self.head = nn.Sequential(conv_block(64, 64), nn.Conv2d(64, 1, 1))

    def forward(self, x):                                 # (B, T, C, H, W)
        B, T = x.shape[:2]
        feats = self.encoder(x.flatten(0, 1)).unflatten(0, (B, T))
        return self.head(self.pool(self.convlstm(feats))).squeeze(1)   # (B, R, C)


# ---------------------------------------------------------------------------
# Dataset + loss
# ---------------------------------------------------------------------------
class FloodCache(Dataset):
    """Cached sample -> (x (T,7,H,W) f16, y (R,C) f32) with lead-time channel."""
    def __init__(self, split):
        import pandas as pd
        m = pd.read_parquet(CACHE_DIR / "manifest.parquet")
        self.dates = [d.strftime("%Y%m%d")
                      for d in m.loc[m.split == split, "label_day"]]

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, i):
        d = self.dates[i]
        x = np.load(CACHE_DIR / f"{d}_x.npy")                    # (T, 6, H, W) f16
        y = np.load(CACHE_DIR / f"{d}_y.npy").astype(np.float32)  # (R, C)
        if USE_TIME:
            t = np.load(CACHE_DIR / f"{d}_t.npy")               # (T,) lead hours
            T, _, H, W = x.shape
            lead = np.empty((T, 1, H, W), np.float16)
            lead[:] = (t / 24.0).astype(np.float16).reshape(T, 1, 1, 1)
            x = np.concatenate([x, lead], axis=1)               # (T, 7, H, W)
        return torch.from_numpy(x), torch.from_numpy(y)


def soft_tversky(logits, targets, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, smooth=1.0):
    """1 - soft Tversky index over the given cells (differentiable backprop loss)."""
    p = torch.sigmoid(logits).float()
    t = targets.float()
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    return 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
