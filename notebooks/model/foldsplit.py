"""K-fold CV over train/val with a single fixed held-out test set (2019-2025).

Standard setup: one **fixed test set** (never trained on, identical for every
fold), and **K-fold cross-validation over the remaining data** for train/val.
Fold k uses one CV block as **val** (early-stopping / threshold) and the other
K-1 blocks as **train**. Every fold's train, val, and the fixed test each span
all 7 years -> matched base rates across splits. Union of the K val blocks =
the entire non-test pool, so every non-test day is validated in exactly one fold.

Blocks are calendar months, assigned by month index so seasons rotate year to
year (12 mod anything != 0), giving all-year / all-season coverage in each split:
  - TEST = months with ``month_index mod TEST_MOD == TEST_OFFSET`` (~1/TEST_MOD,
    default 10 -> ~10% of days, fixed across folds).
  - Non-test months, in chronological order, are dealt round-robin to K CV
    folds. Fold k: val = its block (~ (1-1/TEST_MOD)/K ~= 15% of days at K=6),
    train = the other K-1 blocks (~75%).
  - Buffer: train days within BUFFER_DAYS of any held-out (val/test) day are
    dropped, so no multi-day flood event straddles a train/eval boundary.

Normalization stats and artifacts are fold-keyed by the trainers via
``fold_suffix()`` so each fold's stats come from that fold's train only.

Environment variables (set by the sweep driver):
  FOLD        CV fold 0..N_FOLDS-1 (required; unset raises). The manifest ``split``
              column only marks the CV pool ("cv") vs excluded 2026 ("unused").
  N_FOLDS     number of train/val CV folds (default 6 -> ~15% val, ~75% train).
  TEST_MOD    fixed-test stride (default 10 -> ~10% test).
  TEST_OFFSET which month-class is the fixed test set (default 0).
  BUFFER_DAYS train/eval buffer in days (default 3).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

N_FOLDS = int(os.environ.get("N_FOLDS", 6))
TEST_MOD = int(os.environ.get("TEST_MOD", 10))
TEST_OFFSET = int(os.environ.get("TEST_OFFSET", 0))
BUFFER_DAYS = int(os.environ.get("BUFFER_DAYS", 3))


def get_fold():
    """Current CV fold from the environment, or None for the manifest split."""
    v = os.environ.get("FOLD")
    return int(v) if v not in (None, "") else None


def fold_suffix():
    """Filename tag so per-fold stats/checkpoints don't collide ('' if no fold)."""
    f = get_fold()
    return "" if f is None else f"_f{f}"


def _exists(cache_dir: Path, days):
    """Keep only days whose feature cache is materialized on disk."""
    return [d for d in days if (cache_dir / f"{d}_sum.npy").exists()]


def _month_index(ts: pd.Series) -> pd.Series:
    """Months since 2019-01 (0-based), for the block assignment."""
    return (ts.dt.year - 2019) * 12 + (ts.dt.month - 1)


def test_months():
    """Month indices of the fixed held-out test set (same for every fold)."""
    # any month whose class is TEST_OFFSET; spans all years since 12 % TEST_MOD != 0
    return {mi for mi in range(0, (2025 - 2019 + 1) * 12) if mi % TEST_MOD == TEST_OFFSET}


def fold_splits(cache_dir):
    """Return (train, val, test) YYYYMMDD day-string lists for the current fold.

    ``test`` is the fixed held-out set (identical for all folds); ``val`` is the
    fold's CV block; ``train`` is the other CV blocks minus a buffer. With no
    FOLD set, reproduces the old manifest-``split`` behaviour. Drops 2026 rows
    (marked ``unused`` in the manifest) either way.
    """
    cache_dir = Path(cache_dir)
    m = pd.read_parquet(cache_dir / "manifest.parquet")
    m = m.loc[m.split != "unused"].copy()          # drop truncated-label 2026
    m["label_day"] = pd.to_datetime(m["label_day"])

    fold = get_fold()
    if fold is None:
        raise RuntimeError(
            "FOLD is not set. This project uses blocked K-fold CV — set the FOLD env "
            "var (0..N_FOLDS-1) before calling fold_splits(). run_cv.py sets it per run; "
            "the eval notebooks loop it via fold_days(k)."
        )

    m["mi"] = _month_index(m["label_day"])
    test_mi = test_months()
    is_test = m["mi"].isin(test_mi)

    # --- CV over non-test months: round-robin into N_FOLDS blocks -------------
    non_test_months = sorted(m.loc[~is_test, "mi"].unique())
    val_months = {mi for r, mi in enumerate(non_test_months) if r % N_FOLDS == fold}
    is_val = (~is_test) & m["mi"].isin(val_months)
    is_train = (~is_test) & (~is_val)

    # --- buffer: drop train days near any held-out (val/test) day -------------
    heldout_ord = np.sort(
        m.loc[is_test | is_val, "label_day"].map(pd.Timestamp.toordinal).values
    )
    if len(heldout_ord):
        tr_ord = m.loc[is_train, "label_day"].map(pd.Timestamp.toordinal).values
        pos = np.searchsorted(heldout_ord, tr_ord)
        left = np.where(pos > 0, tr_ord - heldout_ord[np.clip(pos - 1, 0, None)], 10**9)
        right = np.where(
            pos < len(heldout_ord),
            heldout_ord[np.clip(pos, None, len(heldout_ord) - 1)] - tr_ord,
            10**9,
        )
        near = np.minimum(left, right) <= BUFFER_DAYS
        is_train.loc[m.loc[is_train].index[near]] = False

    def days(mask):
        ds = m.loc[mask, "label_day"].dt.strftime("%Y%m%d").tolist()
        return _exists(cache_dir, ds)

    return days(is_train), days(is_val), days(is_test)


def describe(cache_dir):
    """Print a per-split coverage/base-rate table for the current fold (sanity)."""
    cache_dir = Path(cache_dir)
    m = pd.read_parquet(cache_dir / "manifest.parquet")
    m = m.loc[m.split != "unused"].copy()
    m["label_day"] = pd.to_datetime(m["label_day"])
    key = {d.strftime("%Y%m%d"): (d.year, int(n)) for d, n in
           zip(m["label_day"], m["n_pos"])}
    tr, va, te = fold_splits(cache_dir)
    land = 3360
    print(f"fold {get_fold()}/{N_FOLDS}  test_mod={TEST_MOD} buffer={BUFFER_DAYS}d")
    for name, days in (("train", tr), ("val", va), ("test", te)):
        yrs = sorted({key[d][0] for d in days})
        pos = sum(key[d][1] for d in days)
        br = 100 * pos / (len(days) * land) if days else 0.0
        print(f"  {name:5s}  days {len(days):4d}  base {br:5.3f}%  "
              f"years {min(yrs)}-{max(yrs)} ({len(yrs)})")


if __name__ == "__main__":
    import sys

    ROOT = Path(__file__).resolve().parent
    while not (ROOT / "config.py").exists() and ROOT != ROOT.parent:
        ROOT = ROOT.parent
    sys.path.insert(0, str(ROOT))
    from config import CACHE_DIR

    for k in range(N_FOLDS):
        os.environ["FOLD"] = str(k)
        describe(CACHE_DIR)
