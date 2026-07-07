"""Standalone trainer for a simple same-day ConvLSTM flood model (Storm Arthur, June 2026).

Task: from one CDT day's GOES imagery (24 hourly frames, all 16 ABI bands by
default) predict that day's NWS flood-warning grid (per-cell flood probability).

This is the recurrent sibling of cnn3d_arthur_train.py — same task, data, loss, LR
schedule and DDP setup; the only difference is the temporal model. Instead of a
3D CNN convolving over time, a shared 2D CNN encodes each frame and a ConvLSTM
walks the 24 frames, carrying a hidden state that summarises the whole day.

Everything — model, data, hyperparameters, training loop — lives in this one
file. It trains across both GPUs with DistributedDataParallel; launch it with
torchrun (one process per GPU):

    cd StormArthurEvaluation/notebooks
    NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 convlstm_arthur_train.py

(Single-GPU also works: `python convlstm_arthur_train.py`.)

It reuses the day-cache (model-agnostic):
    {CACHE_DIR}/{day}_x.npy   (24, 16, 1500, 2500) float16   GOES cube
    {CACHE_DIR}/{day}_y.npy   (59, 95) uint8                 warning grid
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
from sklearn.metrics import average_precision_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# standalone: the output grid lives in StormArthurEvaluation/siggrid.py (no repo config.py).
SIG_ROOT = Path(__file__).resolve().parent.parent          # StormArthurEvaluation/
sys.path.insert(0, str(SIG_ROOT))
from siggrid import build_grid_cells  # noqa: E402

# ===========================================================================
# Hyperparameters & settings
# ===========================================================================
CACHE_DIR = SIG_ROOT / "cache/storm_arthur"   # shared day-cache

# Which ABI bands to feed, as 1-indexed band numbers (1..16). Default = all 16.
# e.g. BANDS = (2, 3, 7, 10, 13, 16) for the curated set.
BANDS = tuple(range(1, 17))

T_FRAMES = 24                  # hourly GOES frames per CDT day (the cube's time axis)
GRID_R, GRID_C = 59, 95       # output flood grid (CONUS-land 50 km cells)
# Encoder spatial downsample: the encoder shrinks 1500x2500 -> ENC_H x ENC_W, then
# CellPool regrids those pixels onto the 50 km grid. /8 leaves ~10 pixels per cell
# (finer than the grid) so the geographic pooling has enough pixels to average.
ENC_H, ENC_W = 1500 // 8, 2500 // 8           # 187 x 312
HIDDEN = 64                   # ConvLSTM hidden channels (= encoder output channels).
                              # gates conv scales with HIDDEN**2, so this is the main
                              # size lever: 64 -> ~0.5M params, 128 -> ~1.5M.

EPOCHS = 50
BATCH_SIZE = 1                 # per GPU. each cube is ~2.9 GB f16; bump if VRAM allows
WORKERS = 4                    # dataloader processes

# Learning rate: cosine schedule with a short linear warmup. The LR ramps up from
# WARMUP_EPOCHS, then cosine-decays from LR down to MIN_LR over the rest of training
# (warm start avoids early divergence; the decay lets it settle as it converges).
# (A performance-based alternative would be ReduceLROnPlateau on the test metric.)
LR = 3e-4                      # peak LR (reached at the end of warmup)
MIN_LR = 1e-5                 # floor the cosine decays to
WARMUP_EPOCHS = 1            # epochs of linear ramp-up before the cosine decay
# BCE positive weight: floods are ~1.6% of land cells, so up-weight the rare
# positive class or plain BCE collapses to predicting all-zero.
POS_WEIGHT = 50.0
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
    GOES-pixel -> 50 km-cell index. Each cell's feature = mean of the encoder
    pixels that physically fall in it (off-grid pixels are discarded), so the
    regrid respects the GOES->Albers projection instead of just stretching."""

    def __init__(self):
        super().__init__()
        p2c = np.load(CACHE_DIR / "pix2cell_50km.npy")        # full-res (1500, 2500)
        sub = _subsample_index(p2c, ENC_H, ENC_W).reshape(-1)
        n = GRID_R * GRID_C
        idx = torch.from_numpy(sub.astype(np.int64)).clone()
        idx[idx < 0] = n                                      # off-grid -> dump bin
        counts = torch.zeros(n + 1).scatter_add_(0, idx, torch.ones(idx.numel()))
        self.register_buffer("idx", idx)
        self.register_buffer("counts", counts.clamp_min(1.0))
        self.n = n

    def forward(self, x):                                     # (B, C, ENC_H, ENC_W)
        B, C, h, w = x.shape
        out = x.new_zeros(B, C, self.n + 1)
        idx = self.idx.view(1, 1, -1).expand(B, C, -1)
        out.scatter_add_(2, idx, x.reshape(B, C, h * w))
        out = out / self.counts.view(1, 1, -1)
        return out[:, :, :self.n].reshape(B, C, GRID_R, GRID_C)


# ===========================================================================
# Model
# ===========================================================================
class GOESConvLSTM(nn.Module):
    """Per-frame 2D CNN encoder -> ConvLSTM over time -> CellPool regrid -> head.

    Input  : (B, T=24, C=n_bands, 1500, 2500)
    Output : (B, 1, GRID_R, GRID_C) raw logits (apply sigmoid for probability).

    The encoder (shared weights across frames) shrinks each frame to /8 (187x312);
    the ConvLSTM walks the 24 encoded frames; its final hidden state — a summary of
    the whole day — is regridded onto the 50 km grid by CellPool, then a 2D head
    produces the per-cell logit.
    """

    def __init__(self, n_bands, hidden=HIDDEN):
        super().__init__()
        self.hidden = hidden

        # --- per-frame encoder: (B, n_bands, 1500, 2500) -> (B, hidden, 187, 312) ---
        self.encoder = nn.Sequential(
            # normalise the heterogeneous band scales (reflectance vs brightness-temp)
            nn.BatchNorm2d(n_bands),

            nn.Conv2d(n_bands, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 1500x2500 -> 750x1250

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 375x625

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 187x312  (/8 — stop shrinking space)

            nn.Conv2d(128, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),                        # -> 187x312  (no pool; keep /8)
        )

        # --- ConvLSTM gate convolution: from [encoded frame, prev hidden] -> 4 gates ---
        # (input, forget, output, candidate), each `hidden` channels.
        self.gates = nn.Conv2d(hidden + hidden, 4 * hidden, kernel_size=3, padding=1)

        self.pool = CellPool()                # regrid 187x312 pixels -> 59x95 cells

        # --- 2D head: hidden-channel day-summary -> one flood logit per cell ---
        self.head = nn.Sequential(            # runs on the cell grid (59x95)
            nn.Conv2d(hidden, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: B, T, n_bands, 1500, 2500
        B, T = x.shape[:2]

        # encode every frame with the shared 2D CNN
        feats = [self.encoder(x[:, t]) for t in range(T)]    # each: B, hidden, h, w

        # ConvLSTM: walk the frames, carrying hidden state h and cell state c
        h = torch.zeros_like(feats[0])
        c = torch.zeros_like(feats[0])
        for t in range(T):
            gates = self.gates(torch.cat([feats[t], h], dim=1))   # B, 4*hidden, h, w
            i, f, o, g = gates.chunk(4, dim=1)                    # the four LSTM gates
            c = f.sigmoid() * c + i.sigmoid() * g.tanh()          # update cell state
            h = o.sigmoid() * c.tanh()                            # update hidden state

        # h now summarises the whole day -> regrid to cells -> per-cell logit
        out = self.pool(h)                                       # B, hidden, 59, 95
        out = self.head(out)                                     # B, 1, 59, 95
        return out                                              # B, 1, GRID_R, GRID_C


# ===========================================================================
# Data
# ===========================================================================
class SameDayCache(Dataset):
    """One sample per CDT day: GOES cube (selected bands) + warning grid."""

    def __init__(self, days, band_idx):
        self.days = list(days)
        self.band_idx = band_idx              # 0-indexed channels into the 16-band cube

    def __len__(self):
        return len(self.days)

    def __getitem__(self, i):
        d = self.days[i]
        x = np.load(CACHE_DIR / f"{d}_x.npy")             # (24, 16, 1500, 2500) f16
        x = x[:, self.band_idx]                           # (24, n_bands, 1500, 2500)
        y = np.load(CACHE_DIR / f"{d}_y.npy").astype(np.float32)   # (59, 95)
        return torch.from_numpy(x), torch.from_numpy(y)


def load_splits():
    """Read the train/val/test day lists from the cache manifest."""
    m = pd.read_parquet(CACHE_DIR / "manifest.parquet")
    days = lambda s: [d.strftime("%Y%m%d") for d in m.loc[m.split == s, "cst_day"]]  # noqa: E731
    return days("train"), days("val"), days("test")


# ===========================================================================
# Evaluation
# ===========================================================================
@torch.no_grad()
def evaluate(model, days, band_idx, land, dev):
    """Return PR-AUC over land cells for the given days (run on rank 0 only)."""
    model.eval()
    probs, trues = [], []
    loader = DataLoader(SameDayCache(days, band_idx), batch_size=1, num_workers=WORKERS)
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
        dist.init_process_group("nccl", device_id=dev)    # device_id mutes barrier warning
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    is_main = rank == 0

    band_idx = np.array([b - 1 for b in BANDS])           # 1-indexed -> 0-indexed
    _, gr, gc, land = build_grid_cells()                   # land mask (59, 95) bool
    assert (gr, gc) == (GRID_R, GRID_C), f"grid mismatch: {(gr, gc)}"
    land_t = torch.from_numpy(land).to(dev)               # (59, 95) bool

    tr_days, va_days, te_days = load_splits()

    # SyncBatchNorm: per-GPU batch is tiny (1 cube), so sync BN stats across GPUs.
    net = GOESConvLSTM(n_bands=len(BANDS)).to(dev)
    if ddp:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    model = DDP(net, device_ids=[local]) if ddp else net

    if is_main:
        n_params = sum(p.numel() for p in net.parameters())
        print(f"bands={BANDS}  train={len(tr_days)} val={len(va_days)} "
              f"test={len(te_days)}  world={world}  params={n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    # LR schedule: linear warmup -> cosine decay (LR -> MIN_LR over the remaining epochs).
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, total_iters=max(1, WARMUP_EPOCHS))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=MIN_LR)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, [warmup, cosine], milestones=[max(1, WARMUP_EPOCHS)])

    # Loss: BCE-with-logits for now (pos_weight up-weights the rare flood class).
    # TODO: consider focal loss in the future — it down-weights easy negatives and
    # focuses on hard examples, which often helps on this kind of extreme imbalance.
    pos_weight = torch.tensor([POS_WEIGHT], device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    tr_ds = SameDayCache(tr_days, band_idx)
    sampler = DistributedSampler(tr_ds, shuffle=True) if ddp else None
    train_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=(sampler is None),
                              sampler=sampler, num_workers=WORKERS, drop_last=False)

    hist = []                                              # (epoch, lr, train_loss, test_prauc)
    for epoch in range(1, EPOCHS + 1):
        cur_lr = opt.param_groups[0]["lr"]                 # LR used this epoch (pre-step)
        if sampler is not None:
            sampler.set_epoch(epoch)                       # reshuffle each epoch across ranks
        model.train()
        tot = torch.zeros((), device=dev)
        for x, y in train_loader:
            x = x.to(dev).float()                          # (B, 24, n_bands, H, W)
            y = y.to(dev)                                  # (B, 59, 95)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x).squeeze(1)              # (B, 59, 95)
                loss = loss_fn(logits[:, land_t], y[:, land_t])   # land cells only
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.detach()

        if ddp:
            dist.all_reduce(tot)                           # sum losses across ranks
        train_loss = (tot / (len(train_loader) * world)).item()
        scheduler.step()                                   # advance the LR schedule (per epoch)

        if is_main:
            test_prauc = evaluate(net, te_days, band_idx, land, dev)
            hist.append((epoch, cur_lr, train_loss, test_prauc))
            print(f"epoch {epoch:2d}/{EPOCHS}  lr {cur_lr:.2e}  train_loss {train_loss:.4f}  "
                  f"test PR-AUC {test_prauc:.4f}", flush=True)
        if ddp:
            dist.barrier()                                 # keep ranks in lockstep

    if is_main:
        ckpt = SIG_ROOT / "notebooks/convlstm_arthur.pt"
        torch.save(net.state_dict(), ckpt)
        # training history for plotting later (epoch, lr, train_loss, test_prauc)
        np.savez(CACHE_DIR / "convlstm_results.npz", hist=np.array(hist, np.float32))
        print(f"saved {ckpt} + convlstm_results.npz", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
