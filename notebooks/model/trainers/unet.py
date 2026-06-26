"""Model #5 — U-Net encoder/decoder + grid head (GOES -> next-day warning grid).

Task: from CDT day D-1's GOES imagery (8 three-hourly frames, the curated emissive-IR
bands) predict day D's NWS flood-warning grid (per-cell flood probability).

Unlike the time-aware 3D/recurrent models, this treats the day as a single multi-channel
image: the 8 frames are stacked along the channel axis (T*n_ch) and fed to a 2D U-Net.
The encoder downsamples to a /16 bottleneck; the decoder upsamples back with skip
connections (the U-Net hallmark) so fine spatial detail survives. Decoding stops at /4,
CellPool regrids onto the 50 km grid, and a 2D head emits the per-cell logit.

Everything — model, data, loss, LR schedule, DDP — lives in one file. Train across
both GPUs with torchrun (one process per GPU):

    cd notebooks/model
    NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 trainers/unet.py

(Single-GPU also works: `python trainers/unet.py`.)

Data contract (built by 01_prepare_data.ipynb), per CDT day D:
    {CACHE_DIR}/{D:%Y%m%d}_x.npy   (8, 5, 1500, 2500) f16   GOES cube (config.BANDS)
    {CACHE_DIR}/{D:%Y%m%d}_t.npy   (8,) f32                  per-frame lead hours
    {CACHE_DIR}/{D:%Y%m%d}_y.npy   (59, 95) uint8            warning grid (label)
    {CACHE_DIR}/manifest.parquet                             train/val/test split
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("NCCL_P2P_DISABLE", "1")        # PCIe P2P hangs on this box

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# config.py (repo root) = paths/bands/grid; gridindex.build_pix2cell = the cached
# GOES-pixel -> 50 km-cell index (heavy pyproj math kept in one canonical place).
ROOT = Path(__file__).resolve().parent
while not (ROOT / "config.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "notebooks" / "model"   # gridindex.py + outputs/ live here
sys.path.insert(0, str(MODEL_DIR))         # for `gridindex`
OUT_DIR = MODEL_DIR / "outputs"            # model checkpoints + results npz
from gridindex import build_pix2cell  # noqa: E402

from config import BANDS, CACHE_DIR, N_BAND, T_FRAMES, USE_TIME  # noqa: E402

# ===========================================================================
# Hyperparameters & settings
# ===========================================================================
N_CH = N_BAND + (1 if USE_TIME else 0)        # GOES channels per frame (5 bands + lead)
GRID_R, GRID_C = 59, 95                        # output flood grid (CONUS-land 50 km)
# The U-Net decodes back to /4 (vs /8 for the other models): CellPool then regrids those
# 375x625 pixels onto the 50 km grid (~40 px/cell after the skip-rich decode).
ENC_DOWN = 4
ENC_H, ENC_W = 1500 // ENC_DOWN, 2500 // ENC_DOWN          # 375 x 625

EPOCHS = 50
BATCH_SIZE = 10                # per GPU (~55 GB peak measured); fits under 96 GB
WORKERS = 4

LR = 3e-4
MIN_LR = 1e-5
WARMUP_EPOCHS = 1
POS_WEIGHT_CAP = 100.0
SEED = 0


# ===========================================================================
# CellPool — geographically-correct regrid (replaces a blind bilinear resize)
# ===========================================================================
def _subsample_index(p2c, h, w):
    """Nearest-subsample the full-res (1500,2500) pixel->cell index to (h, w)."""
    H, W = p2c.shape
    ri = (np.arange(h) * (H / h) + H / h / 2).astype(np.int64).clip(0, H - 1)
    ci = (np.arange(w) * (W / w) + W / w / 2).astype(np.int64).clip(0, W - 1)
    return p2c[np.ix_(ri, ci)]


class CellPool(nn.Module):
    """Scatter-mean encoder pixels into the (GRID_R, GRID_C) cells via the real
    GOES-pixel -> 50 km-cell index. Each cell's feature = mean of the encoder pixels
    that physically fall in it (off-grid pixels discarded), so the regrid respects the
    GOES->Albers projection instead of just stretching."""

    def __init__(self, pix2cell_sub, grid_r, grid_c):
        super().__init__()
        n = grid_r * grid_c
        idx = torch.from_numpy(pix2cell_sub.reshape(-1).astype(np.int64)).clone()
        idx[idx < 0] = n                                      # off-grid -> dump bin
        counts = torch.zeros(n + 1).scatter_add_(0, idx, torch.ones(idx.numel()))
        self.register_buffer("idx", idx)
        self.register_buffer("counts", counts.clamp_min(1.0))
        self.n, self.grid_r, self.grid_c = n, grid_r, grid_c

    def forward(self, x):                                     # (B, C, ENC_H, ENC_W)
        B, C, h, w = x.shape
        out = x.new_zeros(B, C, self.n + 1)
        idx = self.idx.view(1, 1, -1).expand(B, C, -1)
        out.scatter_add_(2, idx, x.reshape(B, C, h * w))
        out = out / self.counts.view(1, 1, -1)               # per-cell mean
        return out[:, :, :self.n].reshape(B, C, self.grid_r, self.grid_c)


# ===========================================================================
# Model
# ===========================================================================
class GOESUNet(nn.Module):
    """2D U-Net over the day's frames stacked as channels -> CellPool regrid -> head.

    Input  : (B, T=8, C=n_ch, 1500, 2500)
    Output : (B, 1, GRID_R, GRID_C) raw logits (apply sigmoid for probability).

    Each stage below is an explicit two-conv nn.Sequential (read top to bottom). The
    encoder downsamples /1 -> /16 (e1..e4 + bottleneck); the decoder (in forward)
    upsamples back to /4, concatenating the matching encoder feature each step (U-Net
    skip connections). Odd sizes (1500/2500 don't divide by 16) are handled by
    interpolating each decoder map to its skip's exact size before concatenating.
    """

    def __init__(self, pix2cell_sub, grid_r, grid_c, n_ch=N_CH, t=T_FRAMES):
        super().__init__()
        cin = n_ch * t                                  # stack the T frames as channels
        self.inorm = nn.BatchNorm2d(cin)             # normalise the heterogeneous bands
        self.pool = nn.MaxPool2d(2)
        self.e1 = nn.Sequential(                        # /1
            nn.Conv2d(cin, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.e2 = nn.Sequential(                        # /2
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.e3 = nn.Sequential(                        # /4  (skip)
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.e4 = nn.Sequential(                        # /8  (skip)
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.bott = nn.Sequential(                      # /16  bottleneck
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.d4 = nn.Sequential(                        # /8  (up(bott) cat e4)
            nn.Conv2d(256 + 256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.d3 = nn.Sequential(                        # /4  (up(d4) cat e3)
            nn.Conv2d(256 + 128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.cellpool = CellPool(pix2cell_sub, grid_r, grid_c)     # 375x625 -> 59x95
        self.head = nn.Sequential(                      # runs on the cell grid
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):                               # B, T, C, H, W
        B, T, C, H, W = x.shape
        x = self.inorm(x.reshape(B, T * C, H, W))       # B, T*C, H, W
        e1 = self.e1(x)                                 # /1
        e2 = self.e2(self.pool(e1))                     # /2
        e3 = self.e3(self.pool(e2))                     # /4
        e4 = self.e4(self.pool(e3))                     # /8
        b = self.bott(self.pool(e4))                    # /16
        # decoder: upsample to the skip's size, concatenate (skip), conv
        u4 = F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.d4(torch.cat([u4, e4], dim=1))        # /8
        u3 = F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([u3, e3], dim=1))        # /4
        return self.head(self.cellpool(d3))             # B, 1, R, C (logits)


# ===========================================================================
# Data
# ===========================================================================
class FloodCache(Dataset):
    """One CDT day: GOES cube (+ optional lead-time channel) + warning grid."""

    def __init__(self, days):
        self.days = list(days)

    def __len__(self):
        return len(self.days)

    def __getitem__(self, i):
        d = self.days[i]
        x = np.load(CACHE_DIR / f"{d}_x.npy")            # (T, N_BAND, H, W) f16
        y = np.load(CACHE_DIR / f"{d}_y.npy").astype(np.float32)   # (59, 95)
        if USE_TIME:
            t = np.load(CACHE_DIR / f"{d}_t.npy")                  # (T,) lead hours
            T, _, H, W = x.shape
            lead = np.empty((T, 1, H, W), np.float16)
            lead[:] = (t / 24.0).astype(np.float16).reshape(T, 1, 1, 1)
            x = np.concatenate([x, lead], axis=1)                 # (T, N_BAND+1, H, W)
        return torch.from_numpy(x), torch.from_numpy(y)


def load_splits():
    """Read the train/val/test day lists (YYYYMMDD strings) from the manifest."""
    m = pd.read_parquet(CACHE_DIR / "manifest.parquet")
    def days(s):
        return [d.strftime("%Y%m%d") for d in m.loc[m.split == s, "label_day"]]
    return days("train"), days("val"), days("test")


def train_pos_rate(days, land):
    """Mean warning rate over land on the train days -> the BCE positive weight."""
    acc = np.zeros((GRID_R, GRID_C), np.float64)
    for d in days:
        acc += np.load(CACHE_DIR / f"{d}_y.npy")
    return float((acc / len(days))[land].mean())


# ===========================================================================
# Evaluation
# ===========================================================================
@torch.no_grad()
def evaluate(model, days, land, dev):
    """Return PR-AUC over land cells for the given days (run on rank 0 only)."""
    model.eval()
    probs, trues = [], []
    loader = DataLoader(FloodCache(days), batch_size=1, num_workers=WORKERS)
    for x, y in loader:
        x = x.to(dev).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(x)).float().squeeze(1).cpu()   # (1, 59, 95)
        probs.append(p.numpy())
        trues.append(y.numpy())
    probs = np.concatenate(probs)
    trues = np.concatenate(trues)
    return average_precision_score(trues[:, land].ravel(), probs[:, land].ravel())


# ===========================================================================
# Train
# ===========================================================================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")

    # ---- DDP setup (one process per GPU via torchrun; falls back to single-GPU) ----
    ddp = "RANK" in os.environ
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    if ddp:
        dist.init_process_group("nccl", device_id=dev)
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    is_main = rank == 0

    p2c, gr, gc, land = build_pix2cell()                   # cached pixel->cell index
    assert (gr, gc) == (GRID_R, GRID_C), f"grid mismatch: {(gr, gc)}"
    pix_sub = _subsample_index(p2c, ENC_H, ENC_W)
    land_t = torch.from_numpy(land).to(dev)                # (59, 95) bool

    tr_days, va_days, te_days = load_splits()
    tr_eval = tr_days[:64]         # fixed train subset scored each epoch (cheap proxy)
    pos_rate = train_pos_rate(tr_days, land)
    pos_weight = min((1 - pos_rate) / pos_rate, POS_WEIGHT_CAP)

    # SyncBatchNorm: per-GPU batch is tiny (1 cube), so sync BN stats across GPUs.
    net = GOESUNet(pix_sub, gr, gc).to(dev)
    if ddp:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    model = DDP(net, device_ids=[local]) if ddp else net

    if is_main:
        n_params = sum(p.numel() for p in net.parameters())
        print(f"bands={BANDS} use_time={USE_TIME} n_ch={N_CH} "
              f"train={len(tr_days)} val={len(va_days)} test={len(te_days)} "
              f"world={world} params={n_params:,} "
              f"pos_rate={pos_rate:.4f} pos_weight={pos_weight:.1f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, total_iters=max(1, WARMUP_EPOCHS))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=MIN_LR)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, [warmup, cosine], milestones=[max(1, WARMUP_EPOCHS)])

    # Loss: BCE-with-logits; pos_weight up-weights the rare flood class (else it
    # collapses to all-zero).
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=dev))

    tr_ds = FloodCache(tr_days)
    sampler = DistributedSampler(tr_ds, shuffle=True) if ddp else None
    train_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=(sampler is None),
                              sampler=sampler, num_workers=WORKERS, drop_last=False)

    hist = []   # (epoch, lr, loss, train_prauc, val_prauc, test_prauc)
    for epoch in range(1, EPOCHS + 1):
        cur_lr = opt.param_groups[0]["lr"]
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        tot = torch.zeros((), device=dev)
        for x, y in train_loader:
            x = x.to(dev).float()                          # (B, T, n_ch, H, W)
            y = y.to(dev)                                  # (B, 59, 95)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x).squeeze(1)              # (B, 59, 95)
                loss = loss_fn(logits[:, land_t], y[:, land_t])   # land cells only
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.detach()

        if ddp:
            dist.all_reduce(tot)
        train_loss = (tot / (len(train_loader) * world)).item()
        scheduler.step()

        if is_main:
            tr_prauc = evaluate(net, tr_eval, land, dev)
            va_prauc = evaluate(net, va_days, land, dev)
            te_prauc = evaluate(net, te_days, land, dev)
            hist.append((epoch, cur_lr, train_loss, tr_prauc, va_prauc, te_prauc))
            print(f"epoch {epoch:2d}/{EPOCHS}  lr {cur_lr:.2e}  loss {train_loss:.4f}  "
                  f"PR-AUC train {tr_prauc:.4f} val {va_prauc:.4f} test {te_prauc:.4f}",
                  flush=True)
        if ddp:
            dist.barrier()

    if is_main:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt = OUT_DIR / "unet.pt"
        torch.save(net.state_dict(), ckpt)
        np.savez(OUT_DIR / "unet_results.npz", hist=np.array(hist, np.float32))
        print(f"saved {ckpt} + unet_results.npz", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
