"""logreg baseline on the unified GOES cache (config.CACHE_DIR).

Location-free per-cell tabular baseline: linear logistic regression.
59 features/(cell,day) = goes 8*4 + glm 8*2 + daily 3 + lead 8. Imbalance via
class_weight; no image, no spatial-tolerance loss. Saves outputs/logreg.pkl.

Run (single process, NOT torchrun):

    cd notebooks/model && python trainers/logreg.py
"""
import pickle
import sys
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

from config import CACHE_DIR  # noqa: E402

NAME = "logreg"
N_GOES, N_GLM, N_DAILY, T_FRAMES = 4, 2, 3, 8
GRID_R, GRID_C = 59, 95
N_FEAT = T_FRAMES * N_GOES + T_FRAMES * N_GLM + N_DAILY + T_FRAMES   # 59
POS_WEIGHT_CAP = 40.0


def load_splits():
    """Train/val/test day lists, filtered to days whose _img.npy exists on disk."""
    m = pd.read_parquet(CACHE_DIR / "manifest.parquet")

    def days(s):
        ds = [d.strftime("%Y%m%d") for d in m.loc[m.split == s, "label_day"]]
        return [d for d in ds if (CACHE_DIR / f"{d}_img.npy").exists()]

    return days("train"), days("val"), days("test")


def build_xy(days, land):
    """(n_days*n_land, 59) features + (n_days*n_land,) labels over land cells."""
    li, lj = np.where(land)
    X, Y = [], []
    for d in days:
        goes = np.nan_to_num(np.load(CACHE_DIR / f"{d}_goes.npy"))    # (8,4,R,C)
        glm = np.nan_to_num(np.load(CACHE_DIR / f"{d}_glm.npy"))      # (8,2,R,C)
        daily = np.nan_to_num(np.load(CACHE_DIR / f"{d}_daily.npy"))  # (3,R,C)
        t = np.load(CACHE_DIR / f"{d}_t.npy")                         # (8,)
        y = np.load(CACHE_DIR / f"{d}_y.npy")                         # (R,C)
        g = goes[:, :, li, lj].reshape(T_FRAMES * N_GOES, -1).T       # (n_land,32)
        gl = glm[:, :, li, lj].reshape(T_FRAMES * N_GLM, -1).T        # (n_land,16)
        da = daily[:, li, lj].T                                       # (n_land,3)
        le = np.broadcast_to(t, (len(li), T_FRAMES))                  # (n_land,8)
        X.append(np.concatenate([g, gl, da, le], axis=1).astype(np.float32))
        Y.append(y[li, lj].astype(np.int8))
    return np.concatenate(X), np.concatenate(Y)


def fit_model(Xtr, Ytr, Xva, Yva, spw):
    """Standardized linear logistic regression with class weighting."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=200, class_weight={0: 1.0, 1: spw}),
    )
    model.fit(Xtr, Ytr)
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


def _report(model, days, land, label):
    probs, trues = predict_grids({"model": model}, days, land)
    p = probs[:, land].ravel()
    t = trues[:, land].ravel().astype(int)
    ap = average_precision_score(t, p)
    pr, rc, _ = precision_recall_curve(t, p)
    f1 = float(np.nanmax(2 * pr * rc / (pr + rc + 1e-9)))
    base = t.mean()
    print(f"  {label:<5} AUPRC {ap:.4f} ({ap / base:.1f}x base {base:.4f})  "
          f"bestF1 {f1:.3f}", flush=True)


def main():
    _, gr, gc, land = build_pix2cell()
    assert (gr, gc) == (GRID_R, GRID_C)
    tr, va, te = load_splits()
    print(f"[{NAME}] features: train={len(tr)} val={len(va)} test={len(te)} days "
          f"x {int(land.sum())} land cells x {N_FEAT} feats", flush=True)
    Xtr, Ytr = build_xy(tr, land)
    Xva, Yva = build_xy(va, land)
    pos_rate = float(Ytr.mean())
    spw = float(np.clip((1 - pos_rate) / max(pos_rate, 1e-6), 1.0, POS_WEIGHT_CAP))
    print(f"[{NAME}] X={Xtr.shape} pos_rate={pos_rate:.4f} scale_pos_weight={spw:.1f}",
          flush=True)

    model = fit_model(Xtr, Ytr, Xva, Yva, spw)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{NAME}.pkl", "wb") as f:
        pickle.dump({"model": model, "kind": NAME, "n_feat": N_FEAT, "spw": spw}, f)
    print(f"[{NAME}] saved -> {OUT_DIR / (NAME + '.pkl')}", flush=True)

    print(f"[{NAME}] FINAL")
    _report(model, va, land, "val")
    _report(model, te, land, "test")


if __name__ == "__main__":
    main()
