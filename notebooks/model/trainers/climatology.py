"""Baseline #1 — per-cell climatology (no GOES input).

Task: predict day D's NWS flood-warning grid for each 50 km CONUS-land cell. This
baseline ignores the imagery entirely and predicts each cell's **training flood
frequency** — i.e. "how often does this cell get a warning?" — the same map every
day. It is the number every GOES model below must beat: if a model can't out-predict
"floods happen where floods usually happen," its imagery isn't contributing.

There is nothing to learn by gradient descent here (the optimal constant per-cell
predictor *is* the training rate), so this is a closed-form fit — no GPU, no DDP.
Run it directly:

    cd notebooks/model
    python trainers/climatology.py

Data contract (built by 01_prepare_data.ipynb), per CDT day D:
    {CACHE_DIR}/{D:%Y%m%d}_y.npy   (59, 95) uint8   warning grid (label)
    {CACHE_DIR}/manifest.parquet                     train/val/test split
(the _x GOES cube and _t lead times are unused by this baseline.)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# config.py (repo root) is the single source for paths + the output grid.
ROOT = Path(__file__).resolve().parent
while not (ROOT / "config.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "notebooks" / "model"   # outputs/ lives here
OUT_DIR = MODEL_DIR / "outputs"            # model results
from config import CACHE_DIR, build_grid_cells  # noqa: E402

GRID_R, GRID_C = 59, 95               # output flood grid (CONUS-land 50 km cells)
SMOOTH = 1.0   # Laplace smoothing on the per-cell warning rate


# ===========================================================================
# Data
# ===========================================================================
def load_splits():
    """Read the train/val/test day lists (as YYYYMMDD strings) from the manifest."""
    m = pd.read_parquet(CACHE_DIR / "manifest.parquet")
    def days(s):
        return [d.strftime("%Y%m%d") for d in m.loc[m.split == s, "label_day"]]
    return days("train"), days("val"), days("test")


def stack_labels(days):
    """Stack the warning grids for `days` into (N, GRID_R, GRID_C) float32."""
    return np.stack([np.load(CACHE_DIR / f"{d}_y.npy")
                     for d in days]).astype(np.float32)


# ===========================================================================
# Evaluation
# ===========================================================================
def evaluate(pred_map, days, land):
    """PR-AUC and ROC-AUC over land cells, scoring `pred_map` (same map every day)."""
    trues = stack_labels(days)[:, land].ravel()                  # (N * n_land,)
    preds = np.broadcast_to(pred_map[land], (len(days), int(land.sum()))).ravel()
    return (average_precision_score(trues, preds),
            roc_auc_score(trues, preds),
            float(trues.mean()))                                 # base positive rate


# ===========================================================================
# "Fit" (closed form) + report
# ===========================================================================
def main():
    _, gr, gc, land = build_grid_cells()        # land mask (59, 95) bool
    assert (gr, gc) == (GRID_R, GRID_C), f"grid mismatch: {(gr, gc)}"

    tr_days, va_days, te_days = load_splits()

    # per-cell climatology = (smoothed) mean warning frequency over the TRAIN days
    tr_y = stack_labels(tr_days)                                 # (N_train, 59, 95)
    clim = (tr_y.sum(0) + SMOOTH) / (len(tr_days) + 2 * SMOOTH)  # (59, 95) in (0, 1)
    clim[~land] = 0.0

    print(f"train={len(tr_days)} val={len(va_days)} test={len(te_days)}  "
          f"land cells={int(land.sum())}", flush=True)
    print(f"per-cell rate: median {np.median(clim[land]):.4f}  "
          f"max {clim[land].max():.4f}", flush=True)

    for name, days in (("val", va_days), ("test", te_days)):
        prauc, rocauc, base = evaluate(clim, days, land)
        lift = prauc / base if base else float("nan")
        print(f"{name:>4}: PR-AUC {prauc:.4f} ({lift:.1f}x base {base:.4f})  "
              f"ROC-AUC {rocauc:.3f}", flush=True)

    # save the fitted baseline (the climatology map) + a comparable results row
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "climatology.npz"
    te_prauc, te_roc, te_base = evaluate(clim, te_days, land)
    np.savez(out, clim=clim, land=land,
             test_prauc=te_prauc, test_rocauc=te_roc, test_base=te_base)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
