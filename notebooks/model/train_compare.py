"""Train ONE comparison model across BOTH GPUs via DDP; save test predictions.

Part of notebook 02's model comparison. All three models (ConvLSTM / CNN-LSTM /
ResNet, defined in floodnet.py) share the CellPool regrid, the location head, and
the recall-favoring Tversky loss; only the spatio-temporal backbone differs.

Each run does true **DistributedDataParallel** over both cards (balanced — unlike
DataParallel), trains for --epochs, then rank 0 evaluates the temporal test split
and writes probs / truth / training-history to CKPT_DIR/compare/{model}.npz for
the notebook to plot.

Launch (from notebooks/model/):
    torchrun --nproc_per_node=2 train_compare.py --model convlstm --epochs 5
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NCCL_P2P_DISABLE", "1")     # PCIe P2P hangs on this box

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.metrics import average_precision_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parent
while not (ROOT / "config.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))     # for `import floodnet`
from floodnet import (  # noqa: E402
    COMPARE_MODELS,
    FloodCache,
    build_pix2cell,
    pool_sub,
    soft_tversky,
)

from config import (  # noqa: E402
    BATCH_PER_GPU,
    CACHE_DIR,
    CKPT_DIR,
    LR,
    TVERSKY_ALPHA,
    TVERSKY_BETA,
    WORKERS,
)


def temporal_split():
    """Earliest 70% of days -> train, next 10% -> val, last 20% -> test."""
    m = pd.read_parquet(CACHE_DIR / "manifest.parquet")
    days = sorted(d.strftime("%Y%m%d") for d in m["label_day"])
    n = len(days)
    return days[:int(0.70 * n)], days[int(0.70 * n):int(0.80 * n)], days[int(0.80 * n):]


def loader(dates, sampler=None, shuffle=False):
    ds = FloodCache("train"); ds.dates = list(dates)
    return DataLoader(ds, BATCH_PER_GPU, sampler=sampler, shuffle=shuffle,
                      num_workers=WORKERS, pin_memory=True)


@torch.no_grad()
def predict(model, dl, dev, land):
    model.eval()
    P, Y = [], []
    for xb, yb in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(xb.float().to(dev)))[:].float().cpu()
        P.append(p.numpy()); Y.append(yb.numpy())
    return np.concatenate(P), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(COMPARE_MODELS))
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    torch.set_float32_matmul_precision("high")

    if rank == 0:
        build_pix2cell()                               # build/cache the index once
    dist.barrier()
    p2c, gr, gc, land = build_pix2cell()
    sub = pool_sub(p2c)
    mask = torch.from_numpy(land).to(dev)

    tr_dates, va_dates, te_dates = temporal_split()
    clim = np.zeros((gr, gc), np.float32)
    for d in tr_dates:
        clim += np.load(CACHE_DIR / f"{d}_y.npy")
    clim /= len(tr_dates)

    net = COMPARE_MODELS[args.model](sub, gr, gc, clim).to(dev)
    model = DDP(net, device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    tr_ds = FloodCache("train"); tr_ds.dates = list(tr_dates)
    tr_smp = DistributedSampler(tr_ds, shuffle=True)
    tr_dl = DataLoader(tr_ds, BATCH_PER_GPU, sampler=tr_smp,
                       num_workers=WORKERS, pin_memory=True)
    va_dl = loader(va_dates) if rank == 0 else None

    if rank == 0:
        n_par = sum(p.numel() for p in net.parameters())
        print(f"[{args.model}] DDP world={world} | params {n_par:,} | "
              f"train {len(tr_dates)} val {len(va_dates)} test {len(te_dates)} | "
              f"loss Tversky a={TVERSKY_ALPHA} b={TVERSKY_BETA}", flush=True)

    hist = {"epoch": [], "train_loss": [], "val_ap": []}
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        tr_smp.set_epoch(ep)
        model.train()
        tot = torch.zeros((), device=dev)
        for xb, yb in tr_dl:
            xb = xb.to(dev, non_blocking=True).float()
            yb = yb.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = soft_tversky(model(xb)[:, mask], yb[:, mask],
                                    TVERSKY_ALPHA, TVERSKY_BETA)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.detach()
        dist.all_reduce(tot)
        tr_loss = (tot / (len(tr_dl) * world)).item()
        # rank-0 validation (single GPU on the synced weights)
        ap_val = float("nan")
        if rank == 0:
            vp, vy = predict(net, va_dl, dev, land)
            ap_val = average_precision_score(vy[:, land].ravel(), vp[:, land].ravel())
            hist["epoch"].append(ep); hist["train_loss"].append(tr_loss)
            hist["val_ap"].append(ap_val)
            print(f"[{args.model}] epoch {ep}/{args.epochs}  train {tr_loss:.4f}  "
                  f"val PR-AUC {ap_val:.4f}  [{(time.perf_counter()-t0)/60:.1f} min]",
                  flush=True)
        dist.barrier()

    # rank-0 test evaluation + save
    if rank == 0:
        te_dl = loader(te_dates)
        probs, trues = predict(net, te_dl, dev, land)
        out = CKPT_DIR / "compare"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"{args.model}.npz",
                 probs=probs.astype(np.float32), trues=trues.astype(np.uint8),
                 land=land, clim=clim, dates=np.array(te_dates),
                 hist_epoch=np.array(hist["epoch"]),
                 hist_train=np.array(hist["train_loss"]),
                 hist_val=np.array(hist["val_ap"]),
                 params=sum(p.numel() for p in net.parameters()))
        torch.save(net.state_dict(), out / f"{args.model}.pt")
        print(f"[{args.model}] saved {probs.shape} test predictions -> "
              f"{out/(args.model+'.npz')}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
