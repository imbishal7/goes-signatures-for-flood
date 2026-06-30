"""resnet3d trainer on the unified GOES cache (config.CACHE_DIR).

Small 3D-ResNet: stem + 3 residual stages.

Two-branch model: image encoder -> CellPool regrid (59x95) -> fused with the per-cell
GOES/GLM/daily feature branch -> per-cell flood logit. Loss = 0.6 x weighted focal BCE
+ 0.4 x spatial-tolerance. Train across both GPUs with torchrun:

    cd notebooks/model
    NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 trainers/resnet3d.py

(Single-GPU also works.) Writes outputs/<name>.pt + outputs/<name>_results.npz and
prints train/val/test AUPRC + P/R/F1/CSI (exact + 1-grid).
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
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# config.py (repo root) + gridindex (parent dir) = shared 50 km grid index
ROOT = Path(__file__).resolve().parent
while not (ROOT / "config.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "notebooks" / "model"
sys.path.insert(0, str(MODEL_DIR))
OUT_DIR = MODEL_DIR / "outputs"
from gridindex import build_pix2cell  # noqa: E402

from config import CACHE_DIR  # noqa: E402

# ===========================================================================
# Hyperparameters & settings
# ===========================================================================
NAME = "resnet3d"

N_IMG = 7              # image-stack channels (b8,b10,b14 + d10-8,d11-14 + dt_b14,cool)
N_GOES = 4            # per-cell GOES features per frame
N_GLM = 2             # per-cell GLM features per 3h bin
N_DAILY = 3           # per-cell daily summary features
T_FRAMES = 8
GRID_R, GRID_C = 59, 95                        # output flood grid (CONUS-land 50 km)
ENC_DOWN = 8
ENC_H, ENC_W = 750 // ENC_DOWN, 1250 // ENC_DOWN          # 93 x 156 (stored /2 image)

EPOCHS = 50           # quick first pass
BATCH_SIZE = 24         # per GPU
WORKERS = 8

LR = 3e-4
MIN_LR = 1e-5
WARMUP_EPOCHS = 1
GAMMA = 2.0           # focal-loss focusing strength
LOSS_FOCAL_W = 0.6    # weight for weighted focal BCE term
LOSS_STOL_W = 0.4     # weight for spatial tolerance term
POS_WEIGHT_CAP = 40.0
SEED = 0
PATIENCE = 10
WEIGHT_DECAY = 1e-2
DROPOUT = 0.2


# ===========================================================================
# CellPool — geographically-correct regrid (the real GOES-pixel -> 50 km index)
# ===========================================================================
def _subsample_index(p2c, h, w):
    """Nearest-subsample the full-res (1500,2500) pixel->cell index to (h, w)."""
    H, W = p2c.shape
    ri = (np.arange(h) * (H / h) + H / h / 2).astype(np.int64).clip(0, H - 1)
    ci = (np.arange(w) * (W / w) + W / w / 2).astype(np.int64).clip(0, W - 1)
    return p2c[np.ix_(ri, ci)]


class CellPool(nn.Module):
    """Scatter-mean encoder pixels into the (GRID_R, GRID_C) cells via the real
    GOES-pixel -> 50 km-cell index (off-grid pixels discarded)."""

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
# Image encoder (the only model-specific piece) -> (B, ENC_OUT, ~93, ~156)
# ===========================================================================
ENC_OUT = 128


class Res3DBlock(nn.Module):
    """Basic 3D residual block; the stride downsamples residual and skip together."""

    def __init__(self, ci, co, stride):
        super().__init__()
        self.c1 = nn.Conv3d(ci, co, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm3d(co)
        self.c2 = nn.Conv3d(co, co, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm3d(co)
        self.skip = (nn.Sequential(nn.Conv3d(ci, co, 1, stride=stride, bias=False),
                                   nn.BatchNorm3d(co))
                     if (stride != (1, 1, 1) or ci != co) else nn.Identity())

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + self.skip(x))


class Encoder(nn.Module):
    """Small 3D-ResNet: stem + 3 residual stages (one block each)."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.BatchNorm3d(N_IMG),
            nn.Conv3d(N_IMG, 32, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2),
                      bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),        # 8,~375,~625
        )
        self.s1 = Res3DBlock(32, 64, (2, 2, 2))               # 4,~188,~313
        self.s2 = Res3DBlock(64, 128, (2, 2, 2))              # 2,~94,~157
        self.drop = nn.Dropout3d(DROPOUT)
        self.s3 = Res3DBlock(128, 128, (2, 1, 1))             # 1,~94,~157

    def forward(self, img):                                   # (B,T,7,H,W)
        x = self.stem(img.permute(0, 2, 1, 3, 4))
        x = self.s3(self.drop(self.s2(self.s1(x))))
        return x.squeeze(2)                                   # FloodNet adaptive-pools


# ===========================================================================
# Two-branch model — encoder (image) fused with the per-cell feature branch
# ===========================================================================
class FloodNet(nn.Module):
    """Inputs: img (B,T,7,750,1250), goes (B,T,4,R,C), glm (B,T,2,R,C), daily (B,3,R,C),
    t (B,T) lead time. Output: (B,1,R,C) logits (sigmoid for probability)."""

    def __init__(self, pix2cell_sub, grid_r, grid_c):
        super().__init__()
        self.encoder = Encoder()
        self.pool = CellPool(pix2cell_sub, grid_r, grid_c)
        # cell branch: goes + glm (flattened over T) + daily + lead time (T broadcast)
        cell_in = T_FRAMES * N_GOES + T_FRAMES * N_GLM + N_DAILY + T_FRAMES  # 32+16+3+8=59
        self.cell_enc = nn.Sequential(
            nn.BatchNorm2d(cell_in),
            nn.Conv2d(cell_in, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Dropout2d(DROPOUT),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(ENC_OUT + 64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Dropout2d(DROPOUT),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, img, goes, glm, daily, t):
        B = img.shape[0]
        x = self.encoder(img)                                 # (B,ENC_OUT,~93,~156)
        x = F.adaptive_avg_pool2d(x, (ENC_H, ENC_W))          # guarantee CellPool dims
        x = self.pool(x)                                      # (B,ENC_OUT,R,C)
        t_grid = t.view(B, T_FRAMES, 1, 1).expand(B, T_FRAMES, GRID_R, GRID_C)
        cell = torch.cat([goes.reshape(B, -1, GRID_R, GRID_C),
                          glm.reshape(B, -1, GRID_R, GRID_C),
                          daily, t_grid], dim=1)              # (B,59,R,C)
        cell = self.cell_enc(cell)                            # (B,64,R,C)
        return self.head(torch.cat([x, cell], dim=1))        # (B,1,R,C) logits


# ===========================================================================
# Loss — 0.6 x weighted focal BCE + 0.4 x spatial tolerance
# ===========================================================================
def _weighted_focal_bce(logits, y, pos_weight, gamma=2.0):
    """Focal-modulated BCE with scalar pos_weight. logits/y: (B, N_land), fp32."""
    logits = logits.float()
    y = y.float()
    pw = torch.full((1,), pos_weight, dtype=torch.float32, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw, reduction="none")
    p_t = torch.sigmoid(logits) * y + (1 - torch.sigmoid(logits)) * (1 - y)
    return ((1 - p_t) ** gamma * bce).mean()


def _spatial_tolerance_loss(probs, y_f, land_t, eps=1e-6):
    """Spatial-tolerant positive + far-false-alarm loss over the 2D grid.

    Positive: credit_i = max(p[i], 0.5*max_p over 3x3); term = -log(credit_i).
    False alarm: -log(1-p[k]) for land cells with no true flood within 1 grid.
    """
    probs = probs.float()
    y_f = y_f.float()
    max_nbr = F.max_pool2d(probs.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    credit = torch.maximum(probs, 0.5 * max_nbr)
    pos_loss = -(torch.log(credit.clamp(min=eps)) * y_f * land_t)
    pos_norm = (y_f * land_t).sum().clamp(min=1)
    near = F.max_pool2d(y_f.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0
    fa_mask = (~(y_f > 0.5)) & (~near) & land_t.unsqueeze(0)
    fa_loss = -(torch.log((1 - probs).clamp(min=eps)) * fa_mask.float())
    fa_norm = fa_mask.float().sum().clamp(min=1)
    return 0.5 * pos_loss.sum() / pos_norm + 0.5 * fa_loss.sum() / fa_norm


def combined_loss(logits, y, land_t, pos_weight):
    """0.6 x weighted focal BCE (land cells) + 0.4 x spatial tolerance (full grid)."""
    logits_f = logits.float()
    y_f = y.float()
    wfocal = _weighted_focal_bce(logits_f[:, land_t], y_f[:, land_t], pos_weight, GAMMA)
    stol = _spatial_tolerance_loss(torch.sigmoid(logits_f), y_f, land_t)
    return LOSS_FOCAL_W * wfocal + LOSS_STOL_W * stol


# ===========================================================================
# Data
# ===========================================================================
class FeatureCache(Dataset):
    """One CDT day: image stack + per-cell GOES/GLM/daily features + label."""

    def __init__(self, days):
        self.days = list(days)

    def __len__(self):
        return len(self.days)

    def __getitem__(self, i):
        d = self.days[i]
        img = np.load(CACHE_DIR / f"{d}_img.npy")                       # (T,7,H,W) f16
        goes = np.nan_to_num(np.load(CACHE_DIR / f"{d}_goes.npy"))      # (T,4,R,C)
        glm = np.nan_to_num(np.load(CACHE_DIR / f"{d}_glm.npy"))        # (T,2,R,C)
        daily = np.nan_to_num(np.load(CACHE_DIR / f"{d}_daily.npy"))    # (3,R,C)
        t = np.load(CACHE_DIR / f"{d}_t.npy").astype(np.float32)        # (T,) lead time
        y = np.load(CACHE_DIR / f"{d}_y.npy").astype(np.float32)        # (R,C)
        return (torch.from_numpy(img), torch.from_numpy(goes),
                torch.from_numpy(glm), torch.from_numpy(daily),
                torch.from_numpy(t), torch.from_numpy(y))


def load_splits():
    """Train/val/test day lists, filtered to days whose _img.npy exists on disk."""
    m = pd.read_parquet(CACHE_DIR / "manifest.parquet")

    def days(s):
        ds = [d.strftime("%Y%m%d") for d in m.loc[m.split == s, "label_day"]]
        return [d for d in ds if (CACHE_DIR / f"{d}_img.npy").exists()]

    return days("train"), days("val"), days("test")


# ===========================================================================
# Evaluation
# ===========================================================================
def _dilate_1grid(mask):
    """Dilate a 2D bool array by 1 cell in all 8 directions (pure numpy)."""
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    return out


@torch.no_grad()
def evaluate(model, days, land, dev, pos_weight):
    """AUPRC + mean combined loss over land cells (fast per-epoch eval)."""
    model.eval()
    land_t = torch.from_numpy(land).to(dev)
    probs_all, trues_all, losses = [], [], []
    loader = DataLoader(FeatureCache(days), batch_size=1, num_workers=WORKERS)
    for img, goes, glm, daily, t, y in loader:
        img = img.to(dev).float()
        goes = goes.to(dev).float()
        glm = glm.to(dev).float()
        daily = daily.to(dev).float()
        t = t.to(dev).float()
        yt = y.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(img, goes, glm, daily, t).squeeze(1)
        loss = combined_loss(logits, yt, land_t, pos_weight)
        probs_all.append(torch.sigmoid(logits).float().cpu().numpy())
        trues_all.append(y.numpy())
        losses.append(float(loss))
    probs = np.concatenate(probs_all)
    trues = np.concatenate(trues_all)
    prauc = average_precision_score(trues[:, land].ravel(), probs[:, land].ravel())
    return prauc, float(np.mean(losses))


@torch.no_grad()
def evaluate_full(model, days, land, dev, threshold=None):
    """AUPRC + P/R/F1/CSI (exact + 1-grid neighbourhood). Returns a dict."""
    model.eval()
    probs_all, trues_all = [], []
    loader = DataLoader(FeatureCache(days), batch_size=1, num_workers=WORKERS)
    for img, goes, glm, daily, t, y in loader:
        img = img.to(dev).float()
        goes = goes.to(dev).float()
        glm = glm.to(dev).float()
        daily = daily.to(dev).float()
        t = t.to(dev).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(img, goes, glm, daily, t).squeeze(1)
        probs_all.append(torch.sigmoid(logits).float().cpu().numpy())
        trues_all.append(y.numpy())
    probs = np.concatenate(probs_all)
    trues = np.concatenate(trues_all)

    p_land = probs[:, land].ravel()
    t_land = trues[:, land].ravel().astype(np.int32)
    prauc = average_precision_score(t_land, p_land)

    if threshold is None:
        pr_c, rc_c, thr_c = precision_recall_curve(t_land, p_land)
        f1_c = 2 * pr_c * rc_c / (pr_c + rc_c + 1e-9)
        threshold = float(thr_c[np.argmax(f1_c[:-1])])

    yb = (p_land >= threshold).astype(np.int32)
    tp = int((yb * t_land).sum())
    fp = int((yb * (1 - t_land)).sum())
    fn = int(((1 - yb) * t_land).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    csi = tp / (tp + fn + fp + 1e-9)

    h_nb = m_nb = fa_nb = 0
    for i in range(len(probs)):
        yt_2d = (trues[i] > 0.5) & land
        yp_2d = (probs[i] >= threshold) & land
        yt_dil = _dilate_1grid(yt_2d) & land
        yp_dil = _dilate_1grid(yp_2d) & land
        h_nb += int((yt_2d & yp_dil).sum())
        m_nb += int((yt_2d & ~yp_dil).sum())
        fa_nb += int((yp_2d & ~yt_dil).sum())
    prec1 = h_nb / (h_nb + fa_nb + 1e-9)
    rec1 = h_nb / (h_nb + m_nb + 1e-9)
    f1_1 = 2 * prec1 * rec1 / (prec1 + rec1 + 1e-9)
    csi1 = h_nb / (h_nb + m_nb + fa_nb + 1e-9)

    return dict(prauc=prauc, thr=threshold, prec=prec, rec=rec, f1=f1, csi=csi,
                prec1=prec1, rec1=rec1, f1_1=f1_1, csi1=csi1)


# ===========================================================================
# Train
# ===========================================================================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")

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

    p2c, gr, gc, land = build_pix2cell()
    assert (gr, gc) == (GRID_R, GRID_C), f"grid mismatch: {(gr, gc)}"
    pix_sub = _subsample_index(p2c, ENC_H, ENC_W)
    land_t = torch.from_numpy(land).to(dev)

    tr_days, va_days, te_days = load_splits()
    tr_eval = tr_days[:64]

    tr_ys = np.stack([np.load(CACHE_DIR / f"{d}_y.npy") for d in tr_days])
    pos_rate = float(tr_ys[:, land].mean())
    raw_pw = (1 - pos_rate) / max(pos_rate, 1e-6)
    pos_weight = float(np.clip(raw_pw, 1.0, POS_WEIGHT_CAP))

    net = FloodNet(pix_sub, gr, gc).to(dev)
    if ddp:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    model = DDP(net, device_ids=[local]) if ddp else net

    if is_main:
        n_params = sum(p.numel() for p in net.parameters())
        print(
            f"[{NAME}] train={len(tr_days)} val={len(va_days)} test={len(te_days)} "
            f"world={world} params={n_params:,} pos_weight={pos_weight:.1f}",
            flush=True,
        )

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, total_iters=max(1, WARMUP_EPOCHS))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=MIN_LR)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, [warmup, cosine], milestones=[max(1, WARMUP_EPOCHS)])

    tr_ds = FeatureCache(tr_days)
    sampler = DistributedSampler(tr_ds, shuffle=True) if ddp else None
    train_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=(sampler is None),
                              sampler=sampler, num_workers=WORKERS, drop_last=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = OUT_DIR / f"{NAME}.pt"
    hist = []
    best_val, best_epoch, no_improve = -1.0, 0, 0
    for epoch in range(1, EPOCHS + 1):
        cur_lr = opt.param_groups[0]["lr"]
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        tot = torch.zeros((), device=dev)
        for img, goes, glm, daily, t, y in train_loader:
            img, goes = img.to(dev).float(), goes.to(dev).float()
            glm, daily = glm.to(dev).float(), daily.to(dev).float()
            t, y = t.to(dev).float(), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(img, goes, glm, daily, t).squeeze(1)
            loss = combined_loss(logits, y, land_t, pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.detach()

        if ddp:
            dist.all_reduce(tot)
        train_loss = (tot / (len(train_loader) * world)).item()
        scheduler.step()

        stop = torch.zeros(1, device=dev)
        if is_main:
            tr_prauc, _ = evaluate(net, tr_eval, land, dev, pos_weight)
            va_prauc, va_loss = evaluate(net, va_days, land, dev, pos_weight)
            te_prauc, te_loss = evaluate(net, te_days, land, dev, pos_weight)
            hist.append((epoch, cur_lr, train_loss, va_loss, te_loss,
                         tr_prauc, va_prauc, te_prauc))
            improved = va_prauc > best_val
            if improved:
                best_val, best_epoch, no_improve = va_prauc, epoch, 0
                torch.save(net.state_dict(), ckpt)
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    stop[0] = 1.0
            print(
                f"[{NAME}] epoch {epoch:2d}/{EPOCHS}  lr {cur_lr:.2e}  "
                f"loss tr {train_loss:.4f} val {va_loss:.4f}  "
                f"AUPRC tr {tr_prauc:.4f} val {va_prauc:.4f} test {te_prauc:.4f}"
                f"{'  *best' if improved else ''}",
                flush=True,
            )
        if ddp:
            dist.broadcast(stop, src=0)
        if stop.item() > 0:
            if is_main:
                print(f"[{NAME}] early stop @ epoch {epoch} "
                      f"(best val {best_val:.4f} @ {best_epoch})", flush=True)
            break

    if is_main:
        np.savez(OUT_DIR / f"{NAME}_results.npz", hist=np.array(hist, np.float32))
        net.load_state_dict(torch.load(ckpt))
        val_m = evaluate_full(net, va_days, land, dev, threshold=None)
        tr_m = evaluate_full(net, tr_days, land, dev, threshold=val_m["thr"])
        te_m = evaluate_full(net, te_days, land, dev, threshold=val_m["thr"])
        te_y = np.stack([np.load(CACHE_DIR / f"{d}_y.npy") for d in te_days])
        base = float(te_y[:, land].mean())

        def row(label, m):
            return (
                f"  {label:<6}  AUPRC {m['prauc']:.4f}  "
                f"P {m['prec']:.3f}  R {m['rec']:.3f}  "
                f"F1 {m['f1']:.3f}  CSI {m['csi']:.3f}  |  "
                f"P@1 {m['prec1']:.3f}  R@1 {m['rec1']:.3f}  "
                f"F1@1 {m['f1_1']:.3f}  CSI@1 {m['csi1']:.3f}"
            )

        print("=" * 78)
        print(f"[{NAME}] FINAL  best-val @ epoch {best_epoch}  "
              f"threshold {val_m['thr']:.3f} (best-val F1)")
        lift = te_m["prauc"] / base
        print(f"  test base rate {base:.4f}  ->  AUPRC lift {lift:.1f}x")
        print("-" * 78)
        print(row("train", tr_m))
        print(row("val", val_m))
        print(row("test", te_m))
        print("=" * 78, flush=True)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
