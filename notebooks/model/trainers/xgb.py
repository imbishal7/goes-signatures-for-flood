"""xgb baseline on the 50 km GOES/GLM feature grid (config.CACHE_DIR).

Per-cell tabular baseline (pure GOES/GLM signatures): gradient-boosted trees.
168 feats/(cell,day) = seq 8*19 + daily-sum 8 + lead 8. Imbalance via scale_pos_weight;
metrics = AUPRC (+lift), P/R/F1/CSI (exact + 1-grid). Run (NOT torchrun):

    cd notebooks/model && python trainers/xgb.py
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parent
while not (ROOT / "config.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "notebooks" / "model"
sys.path.insert(0, str(MODEL_DIR))
OUT_DIR = MODEL_DIR / "outputs"
from gridindex import build_pix2cell  # noqa: E402
from foldsplit import fold_splits, fold_suffix  # noqa: E402

from config import CACHE_DIR  # noqa: E402

NAME = "xgb"
N_SEQ, N_SUM, T_FRAMES = 19, 8, 8
GRID_R, GRID_C = 59, 95
N_FEAT = T_FRAMES * N_SEQ + N_SUM + T_FRAMES   # 168
POS_WEIGHT = 30.0


def load_splits():
    """Train/val/test day lists for the current CV fold (see foldsplit.py)."""
    return fold_splits(CACHE_DIR)


def build_xy(days, land):
    """(n_days*n_land, 168) feats + labels over land cells (seq+sum+lead)."""
    li, lj = np.where(land)
    X, Y = [], []
    for d in days:
        seq = np.nan_to_num(np.load(CACHE_DIR / f"{d}_seq.npy"))    # (8,19,R,C)
        summ = np.nan_to_num(np.load(CACHE_DIR / f"{d}_sum.npy"))   # (8,R,C)
        t = np.load(CACHE_DIR / f"{d}_t.npy")                       # (8,)
        y = np.load(CACHE_DIR / f"{d}_y.npy")                       # (R,C)
        sq = seq[:, :, li, lj].reshape(T_FRAMES * N_SEQ, -1).T      # (n_land,152)
        sm = summ[:, li, lj].T                                      # (n_land,8)
        le = np.broadcast_to(t, (len(li), T_FRAMES))               # (n_land,8)
        X.append(np.concatenate([sq, sm, le], axis=1).astype(np.float32))
        Y.append(y[li, lj].astype(np.int8))
    return np.concatenate(X), np.concatenate(Y)


def fit_model(Xtr, Ytr, Xva, Yva, spw):
    """Gradient-boosted trees (histogram); validation set for monitoring."""
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise SystemExit("xgboost not installed -- run `uv add xgboost`") from e
    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        scale_pos_weight=spw, eval_metric="aucpr", n_jobs=-1,
    )
    model.fit(Xtr, Ytr, eval_set=[(Xva, Yva)], verbose=False)
    return model


def predict_grids(payload, days, land):
    """Per-cell probabilities -> (N,59,95) grids (off-land = 0) + true labels."""
    li, lj = np.where(land)
    model = payload["model"]
    probs, trues = [], []
    for d in days:
        X, _ = build_xy([d], land)
        p = model.predict_proba(X)[:, 1]
        grid = np.zeros((GRID_R, GRID_C), np.float32)
        grid[li, lj] = p
        probs.append(grid)
        trues.append(np.load(CACHE_DIR / f"{d}_y.npy").astype(np.float32))
    return np.array(probs), np.array(trues)


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


def _metrics(probs, trues, land, threshold=None):
    p = probs[:, land].ravel()
    t = trues[:, land].ravel().astype(int)
    ap = average_precision_score(t, p)
    if threshold is None:
        pr, rc, thr = precision_recall_curve(t, p)
        f1c = 2 * pr * rc / (pr + rc + 1e-9)
        threshold = float(thr[np.argmax(f1c[:-1])])
    yb = (p >= threshold).astype(int)
    tp = int((yb * t).sum())
    fp = int((yb * (1 - t)).sum())
    fn = int(((1 - yb) * t).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    csi = tp / (tp + fn + fp + 1e-9)
    h = mm = fa = 0
    for i in range(len(probs)):
        yt = (trues[i] > 0.5) & land
        yp = (probs[i] >= threshold) & land
        h += int((yt & (_dilate_1grid(yp) & land)).sum())
        mm += int((yt & ~(_dilate_1grid(yp) & land)).sum())
        fa += int((yp & ~(_dilate_1grid(yt) & land)).sum())
    prec1 = h / (h + fa + 1e-9)
    rec1 = h / (h + mm + 1e-9)
    f1_1 = 2 * prec1 * rec1 / (prec1 + rec1 + 1e-9)
    csi1 = h / (h + mm + fa + 1e-9)
    base = float(t.mean())
    return dict(prauc=ap, lift=ap / max(base, 1e-9), thr=threshold, prec=prec, rec=rec,
                f1=f1, csi=csi, f1_1=f1_1, csi1=csi1, base=base)


def main():
    t0 = time.perf_counter()
    _, gr, gc, land = build_pix2cell()
    assert (gr, gc) == (GRID_R, GRID_C)
    tr, va, te = load_splits()
    print(f"[{NAME}] features: train={len(tr)} val={len(va)} test={len(te)} days "
          f"x {int(land.sum())} land cells x {N_FEAT} feats", flush=True)
    Xtr, Ytr = build_xy(tr, land)
    Xva, Yva = build_xy(va, land)
    print(f"[{NAME}] X={Xtr.shape} pos_rate={float(Ytr.mean()):.4f} "
          f"pos_weight={POS_WEIGHT:.0f}", flush=True)

    model = fit_model(Xtr, Ytr, Xva, Yva, POS_WEIGHT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{NAME}{fold_suffix()}.pkl", "wb") as f:
        pickle.dump({"model": model, "kind": NAME, "n_feat": N_FEAT,
                     "spw": POS_WEIGHT}, f)
    print(f"[{NAME}] saved -> {OUT_DIR / (NAME + fold_suffix() + '.pkl')}", flush=True)

    vm = _metrics(*predict_grids({"model": model}, va, land), land, threshold=None)
    tm = _metrics(*predict_grids({"model": model}, te, land), land, threshold=vm["thr"])

    def row(lbl, m):
        return (f"  {lbl:<5} AUPRC {m['prauc']:.4f} ({m['lift']:.1f}x)  "
                f"P {m['prec']:.3f} R {m['rec']:.3f} F1 {m['f1']:.3f} "
                f"CSI {m['csi']:.3f} | F1@1 {m['f1_1']:.3f} CSI@1 {m['csi1']:.3f}")

    el = time.perf_counter() - t0
    print(f"[{NAME}] FINAL  thr {vm['thr']:.3f} (best-val F1)  "
          f"base val {vm['base']:.4f} test {tm['base']:.4f}")
    print(row("val", vm))
    print(row("test", tm))
    print(f"[{NAME}] training time: {int(el // 60)}m {int(el % 60):02d}s", flush=True)


if __name__ == "__main__":
    main()
