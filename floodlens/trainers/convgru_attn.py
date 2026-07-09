"""convgru_attn trainer on the 50 km GOES/GLM feature grid (config.CACHE_DIR).

Per-frame 2D CNN + ConvGRU + temporal attention.
Pure per-cell signatures (no image, no climatology). Inputs: seq (B,8,19,59,95),
daily summaries (B,8,59,95), lead (B,8); output per-cell flood logits (B,1,59,95).
Features are normalized (train-only: log1p on GLM, then standardize) inside the model;
lead time is added as a per-frame channel. Loss = 0.6 focal + 0.3 soft-CSI + 0.1
spatial-tolerance. Train across both GPUs:

    cd notebooks/model
    NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 trainers/convgru_attn.py

Metrics: AUPRC(+lift), P/R/F1/CSI (exact + 1-grid); threshold from val, test last.
"""
import os
import time

os.environ.setdefault("NCCL_P2P_DISABLE", "1")        # PCIe P2P hangs on this box

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from floodlens.config import CACHE_DIR, model_artifact  # noqa: E402
from floodlens.foldsplit import fold_splits, fold_suffix  # noqa: E402
from floodlens.gridindex import build_pix2cell  # noqa: E402

# ===========================================================================
# Hyperparameters
# ===========================================================================
NAME = "convgru_attn"
N_SEQ = 19            # per-cell per-frame features (cache)
N_SUM = 8             # per-cell daily-summary features
SEQ_IN = N_SEQ + 1    # + lead-time channel fed into the encoder
T_FRAMES = 8
GRID_R, GRID_C = 59, 95
SEQ_LOG = (16, 17)    # log1p these seq channels (glm_count, glm_density)
SUM_LOG = (5, 6)      # log1p these sum channels (glm_daily_count, glm_max_3h)

EPOCHS = int(os.environ.get("EPOCHS", 30))
BATCH_SIZE = 10      # per GPU (small batch -> more optimizer steps)
WORKERS = 8
LR = 3e-4
MIN_LR = 1e-5
WARMUP_EPOCHS = 3
GAMMA = 2.0
POS_WEIGHT = 30.0
LOSS_FOCAL_W = 0.6
LOSS_CSI_W = 0.3
LOSS_TOL_W = 0.1
DROPOUT = 0.2
WEIGHT_DECAY = 1e-2
PATIENCE = int(os.environ.get("PATIENCE", 10))
SEED = 0


def gn(c):
    """GroupNorm with 8 groups (falls back to fewer if c not divisible)."""
    g = 8 if c % 8 == 0 else (4 if c % 4 == 0 else 1)
    return nn.GroupNorm(g, c)


def _collapse_time(x):
    """(B,C,T,H,W) -> (B,3C,H,W): concat time mean, max, and last frame."""
    return torch.cat([x.mean(2), x.amax(2), x[:, :, -1]], dim=1)


class ConvGRUCell(nn.Module):
    """One ConvGRU step (lighter than ConvLSTM); conv gates over the spatial grid."""

    def __init__(self, cin, hidden):
        super().__init__()
        self.zr = nn.Conv2d(cin + hidden, 2 * hidden, 3, padding=1)
        self.hh = nn.Conv2d(cin + hidden, hidden, 3, padding=1)

    def forward(self, x, h):
        z, r = torch.chunk(torch.sigmoid(self.zr(torch.cat([x, h], dim=1))), 2, dim=1)
        cand = torch.tanh(self.hh(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * cand



# ===========================================================================
# Encoder: (B, T, SEQ_IN, R, C) -> (B, ENC_OUT, R, C)  (keeps the 59x95 grid)
# ===========================================================================
ENC_OUT = 64


class Encoder(nn.Module):
    """Per-frame 2D CNN -> ConvGRU (all hidden states) -> temporal attention."""

    def __init__(self):
        super().__init__()
        self.frame = nn.Sequential(
            nn.Conv2d(SEQ_IN, 64, 3, padding=1),
            gn(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            gn(64), nn.ReLU(inplace=True),
        )
        self.cell = ConvGRUCell(64, ENC_OUT)
        self.pos = nn.Parameter(torch.zeros(1, T_FRAMES, ENC_OUT, 1, 1))
        self.attn = nn.Conv2d(ENC_OUT, 1, 1)
        self.drop = nn.Dropout2d(DROPOUT)

    def forward(self, seq):                                   # (B,T,SEQ_IN,R,W)
        B, Tn, C, R, W = seq.shape
        x = self.frame(seq.reshape(B * Tn, C, R, W)).view(B, Tn, 64, R, W)
        h = x.new_zeros(B, ENC_OUT, R, W)
        hs = []
        for tt in range(Tn):
            h = self.cell(x[:, tt], h)
            hs.append(h)
        hs = torch.stack(hs, dim=1) + self.pos               # (B,T,ENC_OUT,R,W)
        a = self.attn(hs.reshape(B * Tn, ENC_OUT, R, W)).view(B, Tn, 1, R, W)
        w = torch.softmax(a, dim=1)                          # attention over states
        return self.drop((hs * w).sum(1))                    # (B,ENC_OUT,R,W)


# ===========================================================================
# FloodNet: normalize -> add lead channel -> encoder -> fuse summaries -> logit
# ===========================================================================
class FloodNet(nn.Module):
    """Inputs: seq (B,8,19,R,C), summ (B,8,R,C), t (B,8). Output (B,1,R,C) logits."""

    def __init__(self):
        super().__init__()
        self.register_buffer("seq_mean", torch.zeros(1, 1, N_SEQ, 1, 1))
        self.register_buffer("seq_std", torch.ones(1, 1, N_SEQ, 1, 1))
        self.register_buffer("sum_mean", torch.zeros(1, N_SUM, 1, 1))
        self.register_buffer("sum_std", torch.ones(1, N_SUM, 1, 1))
        self.encoder = Encoder()
        fuse_in = ENC_OUT + N_SUM
        self.head = nn.Sequential(
            nn.Conv2d(fuse_in, 64, 3, padding=1),
            gn(64), nn.ReLU(inplace=True),
            nn.Dropout2d(DROPOUT),
            nn.Conv2d(64, 1, 1),
        )

    def set_stats(self, seq_mean, seq_std, sum_mean, sum_std):
        self.seq_mean.copy_(torch.as_tensor(seq_mean, dtype=torch.float32)
                            .view(1, 1, N_SEQ, 1, 1))
        self.seq_std.copy_(torch.as_tensor(seq_std, dtype=torch.float32)
                           .view(1, 1, N_SEQ, 1, 1))
        self.sum_mean.copy_(torch.as_tensor(sum_mean, dtype=torch.float32)
                            .view(1, N_SUM, 1, 1))
        self.sum_std.copy_(torch.as_tensor(sum_std, dtype=torch.float32)
                           .view(1, N_SUM, 1, 1))

    def _norm(self, seq, summ):
        seq = seq.clone()
        summ = summ.clone()
        for k in SEQ_LOG:
            seq[:, :, k] = torch.log1p(seq[:, :, k].clamp(min=0))
        for k in SUM_LOG:
            summ[:, k] = torch.log1p(summ[:, k].clamp(min=0))
        seq = (seq - self.seq_mean) / self.seq_std
        summ = (summ - self.sum_mean) / self.sum_std
        return seq, summ

    def forward(self, seq, summ, t):
        B = seq.shape[0]
        seq, summ = self._norm(seq.float(), summ.float())
        lead = t.view(B, T_FRAMES, 1, 1, 1).expand(B, T_FRAMES, 1, GRID_R, GRID_C)
        seq = torch.cat([seq, lead], dim=2)                    # (B,T,SEQ_IN,R,C)
        x = self.encoder(seq)                                  # (B,ENC_OUT,R,C)
        fuse = torch.cat([x, summ], dim=1)
        return self.head(fuse)                                 # (B,1,R,C)


# ===========================================================================
# Loss = 0.6 focal + 0.3 soft-CSI + 0.1 spatial-tolerance  (over land cells)
# ===========================================================================
def _weighted_focal_bce(logits, y, pos_weight, gamma=2.0):
    logits = logits.float()
    y = y.float()
    pw = torch.full((1,), pos_weight, dtype=torch.float32, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * y + (1 - p) * (1 - y)
    return ((1 - p_t) ** gamma * bce).mean()


def _soft_csi_loss(probs, y, eps=1.0):
    """1 - soft CSI = 1 - TP / (TP + FP + FN), TP/FP/FN from probabilities."""
    probs = probs.float()
    y = y.float()
    tp = (probs * y).sum()
    fp = (probs * (1 - y)).sum()
    fn = ((1 - probs) * y).sum()
    return 1.0 - tp / (tp + fp + fn + eps)


def _spatial_tolerance_loss(probs, y, land_t, eps=1e-6):
    """Credit true cells for prob within 1 grid: -log(max(p, 0.5*max_nbr))."""
    probs = probs.float()
    y = y.float()
    max_nbr = F.max_pool2d(probs.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    credit = torch.maximum(probs, 0.5 * max_nbr)
    pos = y * land_t                                           # true flood on land
    loss = -(torch.log(credit.clamp(min=eps)) * pos)
    return loss.sum() / pos.sum().clamp(min=1)


def combined_loss(logits, y, land_t, pos_weight):
    logits = logits.float()
    probs = torch.sigmoid(logits)
    lf = logits[:, land_t]
    yf = y.float()[:, land_t]
    focal = _weighted_focal_bce(lf, yf, pos_weight, GAMMA)
    csi = _soft_csi_loss(probs[:, land_t], yf)
    tol = _spatial_tolerance_loss(probs, y.float(), land_t)
    return LOSS_FOCAL_W * focal + LOSS_CSI_W * csi + LOSS_TOL_W * tol


# ===========================================================================
# Data + train-only feature stats
# ===========================================================================
class FeatureCache(Dataset):
    """One CDT day: seq grid + daily summaries + lead + label."""

    def __init__(self, days):
        self.days = list(days)

    def __len__(self):
        return len(self.days)

    def __getitem__(self, i):
        d = self.days[i]
        seq = np.nan_to_num(np.load(CACHE_DIR / f"{d}_seq.npy"))    # (8,19,R,C)
        summ = np.nan_to_num(np.load(CACHE_DIR / f"{d}_sum.npy"))   # (8,R,C)
        t = np.load(CACHE_DIR / f"{d}_t.npy").astype(np.float32)    # (8,)
        y = np.load(CACHE_DIR / f"{d}_y.npy").astype(np.float32)    # (R,C)
        return (torch.from_numpy(seq), torch.from_numpy(summ),
                torch.from_numpy(t), torch.from_numpy(y))


def load_splits():
    """Train/val/test day lists for the current CV fold (see foldsplit.py)."""
    return fold_splits(CACHE_DIR)


def load_or_compute_stats(days, land):
    """Train-only per-channel mean/std (log1p GLM) over land; cached to disk."""
    fp = CACHE_DIR / f"feat_stats{fold_suffix()}.npz"
    if fp.exists():
        z = np.load(fp)
        return z["seq_mean"], z["seq_std"], z["sum_mean"], z["sum_std"]
    li, lj = np.where(land)
    ns = np.zeros(N_SEQ)
    nss = np.zeros(N_SEQ)
    cs = 0
    us = np.zeros(N_SUM)
    uss = np.zeros(N_SUM)
    cu = 0
    for d in days:
        seq = np.nan_to_num(np.load(CACHE_DIR / f"{d}_seq.npy")).astype(np.float64)
        summ = np.nan_to_num(np.load(CACHE_DIR / f"{d}_sum.npy")).astype(np.float64)
        for k in SEQ_LOG:
            seq[:, k] = np.log1p(np.clip(seq[:, k], 0, None))
        for k in SUM_LOG:
            summ[k] = np.log1p(np.clip(summ[k], 0, None))
        v = seq[:, :, li, lj]                                  # (8,19,nland)
        ns += v.sum((0, 2))
        nss += (v ** 2).sum((0, 2))
        cs += v.shape[0] * v.shape[2]
        w = summ[:, li, lj]                                    # (8,nland)
        us += w.sum(1)
        uss += (w ** 2).sum(1)
        cu += w.shape[1]
    seq_mean = ns / cs
    seq_std = np.sqrt(np.maximum(nss / cs - seq_mean ** 2, 1e-12)) + 1e-6
    sum_mean = us / cu
    sum_std = np.sqrt(np.maximum(uss / cu - sum_mean ** 2, 1e-12)) + 1e-6
    np.savez(fp, seq_mean=seq_mean, seq_std=seq_std, sum_mean=sum_mean, sum_std=sum_std)
    return seq_mean, seq_std, sum_mean, sum_std


# ===========================================================================
# Evaluation (AUPRC primary; P/R/F1/CSI exact + 1-grid at the val threshold)
# ===========================================================================
def _dilate_1grid(mask):
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
def _collect(model, days, dev):
    model.eval()
    probs, trues = [], []
    loader = DataLoader(FeatureCache(days), batch_size=16, num_workers=WORKERS)
    for seq, summ, t, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(seq.to(dev).float(), summ.to(dev).float(),
                           t.to(dev).float()).squeeze(1)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(probs), np.concatenate(trues)


@torch.no_grad()
def evaluate(model, days, land, dev):
    """AUPRC over land cells (fast per-epoch eval)."""
    probs, trues = _collect(model, days, dev)
    return average_precision_score(trues[:, land].ravel(), probs[:, land].ravel())


@torch.no_grad()
def evaluate_full(model, days, land, dev, threshold=None):
    """AUPRC(+lift), P/R/F1/CSI (exact + 1-grid) at threshold (best-val F1 if None)."""
    probs, trues = _collect(model, days, dev)
    p = probs[:, land].ravel()
    t = trues[:, land].ravel().astype(np.int32)
    prauc = average_precision_score(t, p)
    if threshold is None:
        pr, rc, thr = precision_recall_curve(t, p)
        f1c = 2 * pr * rc / (pr + rc + 1e-9)
        threshold = float(thr[np.argmax(f1c[:-1])])
    yb = (p >= threshold).astype(np.int32)
    tp = int((yb * t).sum())
    fp = int((yb * (1 - t)).sum())
    fn = int(((1 - yb) * t).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    csi = tp / (tp + fn + fp + 1e-9)
    h = m = fa = 0
    for i in range(len(probs)):
        yt = (trues[i] > 0.5) & land
        yp = (probs[i] >= threshold) & land
        h += int((yt & (_dilate_1grid(yp) & land)).sum())
        m += int((yt & ~(_dilate_1grid(yp) & land)).sum())
        fa += int((yp & ~(_dilate_1grid(yt) & land)).sum())
    prec1 = h / (h + fa + 1e-9)
    rec1 = h / (h + m + 1e-9)
    f1_1 = 2 * prec1 * rec1 / (prec1 + rec1 + 1e-9)
    csi1 = h / (h + m + fa + 1e-9)
    base = float(t.mean())
    return dict(prauc=prauc, lift=prauc / max(base, 1e-9), thr=threshold, prec=prec,
                rec=rec, f1=f1, csi=csi, f1_1=f1_1, csi1=csi1, base=base)


# ===========================================================================
# Train (DDP via torchrun; single-GPU also works)
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

    _, gr, gc, land = build_pix2cell()
    assert (gr, gc) == (GRID_R, GRID_C), f"grid mismatch: {(gr, gc)}"
    land_t = torch.from_numpy(land).to(dev)

    tr_days, va_days, te_days = load_splits()
    tr_eval = tr_days[:64]

    if is_main:
        load_or_compute_stats(tr_days, land)                  # ensure stats file exists
    if ddp:
        dist.barrier()
    stats = load_or_compute_stats(tr_days, land)

    net = FloodNet()
    net.set_stats(*stats)
    net = net.to(dev)
    model = DDP(net, device_ids=[local]) if ddp else net      # GroupNorm -> no SyncBN

    if is_main:
        n_params = sum(p.numel() for p in net.parameters())
        print(f"[{NAME}] train={len(tr_days)} val={len(va_days)} test={len(te_days)} "
              f"world={world} params={n_params:,} bs={BATCH_SIZE} pw={POS_WEIGHT:.0f}",
              flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1,
                                               total_iters=max(1, WARMUP_EPOCHS))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=MIN_LR)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, [warmup, cosine], milestones=[max(1, WARMUP_EPOCHS)])

    tr_ds = FeatureCache(tr_days)
    sampler = DistributedSampler(tr_ds, shuffle=True) if ddp else None
    train_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=(sampler is None),
                              sampler=sampler, num_workers=WORKERS, drop_last=False)

    ckpt = model_artifact(NAME, fold_suffix(), "pt")
    hist = []
    best_val, best_epoch, no_improve = -1.0, 0, 0
    t_train0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        cur_lr = opt.param_groups[0]["lr"]
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        tot = torch.zeros((), device=dev)
        for seq, summ, t, y in train_loader:
            seq = seq.to(dev).float()
            summ = summ.to(dev).float()
            t = t.to(dev).float()
            y = y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(seq, summ, t).squeeze(1)
            loss = combined_loss(logits, y, land_t, POS_WEIGHT)
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
            tr_pr = evaluate(net, tr_eval, land, dev)          # monitor only
            va_pr = evaluate(net, va_days, land, dev)          # model selection
            hist.append((epoch, cur_lr, train_loss, tr_pr, va_pr))
            improved = va_pr > best_val
            if improved:
                best_val, best_epoch, no_improve = va_pr, epoch, 0
                torch.save(net.state_dict(), ckpt)
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    stop[0] = 1.0
            print(f"[{NAME}] ep {epoch:2d}/{EPOCHS} lr {cur_lr:.1e} "
                  f"loss {train_loss:.4f} AUPRC tr {tr_pr:.4f} val {va_pr:.4f}"
                  f"{'  *best' if improved else ''}", flush=True)
        if ddp:
            dist.broadcast(stop, src=0)
        if stop.item() > 0:
            if is_main:
                print(f"[{NAME}] early stop @ {epoch} "
                      f"(best val {best_val:.4f} @ {best_epoch})", flush=True)
            break

    if is_main:
        np.savez(model_artifact(NAME, f"{fold_suffix()}_results", "npz"),
                 hist=np.array(hist, np.float32))
        net.load_state_dict(torch.load(ckpt))
        vm = evaluate_full(net, va_days, land, dev, threshold=None)
        tm = evaluate_full(net, te_days, land, dev, threshold=vm["thr"])  # final test
        el = time.perf_counter() - t_train0

        def row(lbl, mtr):
            return (f"  {lbl:<5} AUPRC {mtr['prauc']:.4f} ({mtr['lift']:.1f}x)  "
                    f"P {mtr['prec']:.3f}  R {mtr['rec']:.3f}  "
                    f"F1 {mtr['f1']:.3f}  CSI {mtr['csi']:.3f}  |  "
                    f"F1@1 {mtr['f1_1']:.3f}  CSI@1 {mtr['csi1']:.3f}")

        print("=" * 78)
        print(f"[{NAME}] FINAL  best-val @ epoch {best_epoch}  "
              f"thr {vm['thr']:.3f} (best-val F1)")
        print(f"  base rate: val {vm['base']:.4f}  test {tm['base']:.4f}")
        print(row("val", vm))
        print(row("test", tm))
        print(f"[{NAME}] training time: {int(el // 60)}m {int(el % 60):02d}s "
              f"({epoch} epochs run)")
        print("=" * 78, flush=True)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
