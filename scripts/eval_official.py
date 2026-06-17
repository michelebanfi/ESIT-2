"""Evaluate checkpoints under the MANDATORY official scheme (2026-06-17 clarification).

The organizers fixed the detector to the competition_notebook.ipynb pipeline:
  single fine-tuned model -> its OWN full-train errors (vs original labels)
  -> LogisticRegression() (default) -> test errors (vs task2_test_positions.npy)
  -> predict() at 0.5 -> id,is_forget CSV.

This is the only sanctioned evaluation now. GMM / OOF / thresholds are not allowed in
the submission path. The right offline accuracy proxy is therefore the agreement of the
*official LR's* test predictions with the exp06 direct-classifier pseudo-labels
(~96% CV), NOT the GMM agreement that eval_robust.py reports.

For each --ckpt we report:
  - official-LR train self-accuracy (the notebook's "Logistic Regression Accuracy")
  - official-LR test forget rate  (want ~0.50; >>0.50 = over-prediction / transfer gap)
  - agreement of official-LR test preds with exp06 pseudo-labels  (Kaggle-acc proxy)
  - GMM test forget rate + GMM/pseudo agreement, for reference only

Usage:
  python scripts/eval_official.py --ckpt A.pth [--ckpt B.pth ...] [--out-dir submissions/]
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import load_model
from src.dataset import format_csi_for_cnn
from src.metrics import (get_predictions, prediction_errors, mia_accuracy,
                         gmm_threshold_predictions)

DATA = ROOT / "data" / "public"
PSEUDO_PATH = ROOT / "experiments" / "exp06_direct_classifier" / "test_proba.npy"

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", action="append", required=True,
                    help="checkpoint path; repeat for several models to compare")
parser.add_argument("--out-dir", default=None,
                    help="if set, write a compliant submission CSV per checkpoint here")
args = parser.parse_args()

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

print("Loading data...")
csi_tr = np.load(DATA / "task2_train_csi.npy")
pos_tr = np.load(DATA / "task2_train_positions.npy")[:, :2]
meta_tr = pd.read_csv(DATA / "task2_train_metadata.csv")
y_tr = meta_tr["is_forget"].values

csi_te = np.load(DATA / "task2_test_csi.npy")
pos_te = np.load(ROOT / "data" / "task2_test_positions.npy")[:, :2]
meta_te = pd.read_csv(DATA / "task2_test_metadata.csv")

X_tr = format_csi_for_cnn(csi_tr)
X_te = format_csi_for_cnn(csi_te)
pseudo = (np.load(PSEUDO_PATH) > 0.5).astype(int) if PSEUDO_PATH.exists() else None

if args.out_dir:
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

header = (f"{'checkpoint':<48} {'trainLR':>8} {'retainE':>8} {'forgetE':>8} "
          f"{'LRfgt%':>7} {'LR~acc':>7} {'GMMfgt%':>8} {'GMM~acc':>8}")
print("\n" + header)
print("-" * len(header))

for ckpt in args.ckpt:
    model = load_model(ckpt, DEVICE)
    err_tr = prediction_errors(get_predictions(model, X_tr, device=DEVICE), pos_tr)
    err_te = prediction_errors(get_predictions(model, X_te, device=DEVICE), pos_te)

    # --- official notebook LR: fit on this model's train errors, predict test ---
    mia = mia_accuracy(err_tr, y_tr, err_tr, y_tr)        # train self-accuracy
    lr_test = mia["lr_model"].predict(err_te.reshape(-1, 1)).astype(int)
    lr_fgt = float(lr_test.mean())
    lr_acc = float((lr_test == pseudo).mean()) if pseudo is not None else float("nan")

    # --- GMM, reference only (NOT submittable) ---
    gmm = gmm_threshold_predictions(err_te)
    gmm_fgt = gmm["forget_rate"]
    gmm_acc = float((gmm["predictions"] == pseudo).mean()) if pseudo is not None else float("nan")

    name = Path(ckpt).parent.name + "/" + Path(ckpt).name
    print(f"{name:<48} {mia['mia_accuracy']:>8.4f} {err_tr[y_tr==0].mean():>8.4f} "
          f"{err_tr[y_tr==1].mean():>8.4f} {lr_fgt:>7.3f} {lr_acc:>7.4f} "
          f"{gmm_fgt:>8.3f} {gmm_acc:>8.4f}")

    if args.out_dir:
        out = Path(args.out_dir) / f"submission_{Path(ckpt).parent.name}.csv"
        pd.DataFrame({"id": meta_te["sample_index"], "is_forget": lr_test}).to_csv(out, index=False)
        print(f"    -> wrote {out}")

print("\nLR~acc = official-LR test preds vs exp06 pseudo-labels (the relevant proxy now).")
print("LRfgt% near 0.50 is healthy; >>0.50 = train-fitted threshold over-predicts on test.")
