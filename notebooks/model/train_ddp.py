"""DDP training for FloodConvLSTM — both GPUs, bf16. Training only.

Launch from the notebook (or a shell), cwd = notebooks/model::

    torchrun --nproc_per_node=2 train_ddp.py
    SMOKE=1 torchrun --nproc_per_node=2 train_ddp.py   # quick 1-epoch subset

Saves the best (lowest val loss) checkpoint to CKPT_DIR/floodconvlstm_best.pt.
Metrics + predictions are done back in notebook 02 (which loads this checkpoint).
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NCCL_P2P_DISABLE", "1")     # PCIe P2P hangs on this box

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

ROOT = Path(__file__).resolve().parent
while not (ROOT / "config.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))     # for `import floodnet`
from floodnet import (  # noqa: E402
    FloodCache,
    FloodConvLSTM,
    build_pix2cell,
    pool_sub,
    soft_tversky,
)

from config import BATCH_PER_GPU, CKPT_DIR, EPOCHS, LR, WORKERS  # noqa: E402

SMOKE = bool(os.environ.get("SMOKE"))


def run_epoch(model, dl, mask, opt, dev, train):
    model.train(train)
    total = torch.zeros((), device=dev)
    for xb, yb in dl:
        xb = xb.to(dev, non_blocking=True).float()
        yb = yb.to(dev, non_blocking=True)
        with torch.set_grad_enabled(train), \
                torch.autocast("cuda", dtype=torch.bfloat16):
            loss = soft_tversky(model(xb)[:, mask], yb[:, mask])
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total += loss.detach()
    return total


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)

    # build (or load) the pixel->cell index once on rank 0, then everyone loads it
    if rank == 0:
        build_pix2cell()
    dist.barrier()
    pix2cell, grid_r, grid_c, land = build_pix2cell()
    sub = pool_sub(pix2cell)
    mask = torch.from_numpy(land).to(dev)

    net = FloodConvLSTM(sub, grid_r, grid_c).to(dev)
    model = DDP(net, device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    tr_ds, va_ds = FloodCache("train"), FloodCache("val")
    epochs = EPOCHS
    if SMOKE:
        tr_ds, va_ds, epochs = Subset(tr_ds, range(4)), Subset(va_ds, range(2)), 1
    tr_smp = DistributedSampler(tr_ds, shuffle=True)
    va_smp = DistributedSampler(va_ds, shuffle=False)
    nw = 2 if SMOKE else WORKERS
    tr_dl = DataLoader(tr_ds, BATCH_PER_GPU, sampler=tr_smp,
                       num_workers=nw, pin_memory=True)
    va_dl = DataLoader(va_ds, BATCH_PER_GPU, sampler=va_smp,
                       num_workers=2, pin_memory=True)

    if rank == 0:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"DDP world={world} | train {len(tr_ds)} val {len(va_ds)} | "
              f"epochs {epochs} | grid {grid_r}x{grid_c}", flush=True)

    best, t0 = float("inf"), time.perf_counter()
    for ep in range(1, epochs + 1):
        tr_smp.set_epoch(ep)
        tsum = run_epoch(model, tr_dl, mask, opt, dev, True)
        with torch.no_grad():
            vsum = run_epoch(model, va_dl, mask, opt, dev, False)
        dist.all_reduce(tsum)
        dist.all_reduce(vsum)
        tr_loss = (tsum / (len(tr_dl) * world)).item()
        va_loss = (vsum / (len(va_dl) * world)).item()
        if rank == 0:
            tag = ""
            if va_loss < best:
                best = va_loss
                torch.save(net.state_dict(), CKPT_DIR / "floodconvlstm_best.pt")
                tag = "  <- best (saved)"
            print(f"epoch {ep}/{epochs}  train {tr_loss:.4f}  val {va_loss:.4f}"
                  f"  [{(time.perf_counter()-t0)/60:.1f} min]{tag}", flush=True)
    if rank == 0:
        print(f"done in {(time.perf_counter()-t0)/60:.1f} min  | "
              f"best val {best:.4f} -> {CKPT_DIR/'floodconvlstm_best.pt'}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
