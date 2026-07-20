"""Drive the blocked K-fold CV sweep: train every model on every fold.

Deep models train across both GPUs (DDP via torchrun, nproc=2); tabular models
run single-process. Each (model, fold) writes fold-keyed artifacts to its own
per-model subdir ``outputs/<name>/`` (``<name>_f<k>.pt`` / ``.pkl`` + deep
``<name>_f<k>_results.npz``, via ``config.model_artifact``), so the run is
**resumable** — an existing artifact is skipped unless ``--force``. Fold splits
and the leakage-safe per-fold normalization stats come from floodlens.foldsplit.

    # from the repo root:
    python -m floodlens.run_cv                              # full sweep: 4 models x 6 folds
    python -m floodlens.run_cv --models xgb --folds 0 1     # sanity subset
    python -m floodlens.run_cv --epochs 2 --models resnet3d --folds 0  # deep smoke test
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from floodlens.config import model_artifact

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                    # repo root — run trainers as -m floodlens.trainers.*

DEEP = ["resnet3d", "cnn_attn", "convgru_attn", "convgru_attn_tversky"]
TAB = ["xgb"]
ALL = DEEP + TAB


def artifact(model: str, fold: int) -> Path:
    ext = "pt" if model in DEEP else "pkl"
    return model_artifact(model, f"_f{fold}", ext, make=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--folds", nargs="+", type=int,
                    default=list(range(int(os.environ.get("N_FOLDS", 6)))))
    ap.add_argument("--epochs", type=int, default=None,
                    help="override EPOCHS for deep trainers (smoke tests)")
    ap.add_argument("--force", action="store_true", help="retrain even if present")
    args = ap.parse_args()

    jobs = [(m, f) for m in args.models for f in args.folds]      # model-major
    print(f"CV sweep: {len(jobs)} runs "
          f"({len(args.models)} models x {len(args.folds)} folds)", flush=True)

    done = skipped = failed = 0
    t0 = time.perf_counter()
    for i, (model, fold) in enumerate(jobs, 1):
        out = artifact(model, fold)
        tag = f"[{i}/{len(jobs)}] {model} fold {fold}"
        if out.exists() and not args.force:
            print(f"{tag}: SKIP (exists)", flush=True)
            skipped += 1
            continue

        env = dict(os.environ, FOLD=str(fold), NCCL_P2P_DISABLE="1")
        if args.epochs is not None:
            env["EPOCHS"] = str(args.epochs)
        if model in DEEP:
            # torchrun (torch.distributed.run) launching the trainer as a module
            cmd = [sys.executable, "-m", "torch.distributed.run",
                   "--nproc_per_node=2", "-m", f"floodlens.trainers.{model}"]
        else:
            cmd = [sys.executable, "-m", f"floodlens.trainers.{model}"]

        print(f"{tag}: RUN  {' '.join(cmd)}", flush=True)
        ts = time.perf_counter()
        r = subprocess.run(cmd, cwd=REPO, env=env)
        dt = time.perf_counter() - ts
        if r.returncode == 0 and out.exists():
            print(f"{tag}: OK ({dt/60:.1f} min)", flush=True)
            done += 1
        else:
            print(f"{tag}: FAIL rc={r.returncode} ({dt/60:.1f} min)", flush=True)
            failed += 1

    print(f"\nsweep done in {(time.perf_counter()-t0)/60:.1f} min  "
          f"ok={done} skip={skipped} fail={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
