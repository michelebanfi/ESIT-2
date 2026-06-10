"""Robust evaluation of an unlearning checkpoint.

The naive pipeline fits LR on train errors and applies it to test errors, but the
model memorises train so train errors are unrepresentatively small — the threshold
does not transfer. This script evaluates a checkpoint three ways:

  1. self-MIA on train errors (legacy metric, for changelog continuity)
  2. unsupervised GMM split of *test* errors -> is_forget predictions + forget rate
  3. agreement of those predictions with the exp06 direct classifier's test
     predictions (96% CV accuracy -> good pseudo-labels for offline scoring)

Usage:
  python scripts/eval_robust.py --ckpt experiments/exp01_finetune_retain/model_best.pth
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from src.model import load_model
from src.dataset import format_csi_for_cnn
from src.metrics import (get_predictions, prediction_errors, localization_stats,
                         mia_accuracy, gmm_threshold_predictions)

DATA = ROOT / "data" / "public"
PSEUDO_PATH = ROOT / "experiments" / "exp06_direct_classifier" / "test_proba.npy"

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--out", default=None, help="optional path for submission CSV")
args = parser.parse_args()

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = load_model(args.ckpt, DEVICE)
model.eval()

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

print("Computing errors...")
err_tr = prediction_errors(get_predictions(model, X_tr, device=DEVICE), pos_tr)
err_te = prediction_errors(get_predictions(model, X_te, device=DEVICE), pos_te)

results = {"ckpt": args.ckpt}

# 1. legacy self-MIA
mia = mia_accuracy(err_tr, y_tr, err_tr, y_tr)
results["mia_train_self_accuracy"] = mia["mia_accuracy"]
results["retain"] = localization_stats(err_tr[y_tr == 0])
results["forget"] = localization_stats(err_tr[y_tr == 1])
results["test"] = localization_stats(err_te)
print(f"  self-MIA (train) = {mia['mia_accuracy']:.4f}")
print(f"  retain err = {results['retain']['mean_m']:.4f}m   "
      f"forget err = {results['forget']['mean_m']:.4f}m   "
      f"test err = {results['test']['mean_m']:.4f}m")

# 2. unsupervised GMM split of test errors
gmm = gmm_threshold_predictions(err_te)
preds_te = gmm["predictions"]
results["gmm_test_forget_rate"] = gmm["forget_rate"]
results["gmm_component_means_m"] = gmm["component_means_m"]
print(f"  GMM test split: forget rate = {gmm['forget_rate']:.4f}  "
      f"component mean errors = {[f'{m:.3f}' for m in gmm['component_means_m']]} m")

# 3. agreement with direct-classifier pseudo-labels
if PSEUDO_PATH.exists():
    pseudo = (np.load(PSEUDO_PATH) > 0.5).astype(int)
    agree = float((preds_te == pseudo).mean())
    results["agreement_with_direct_classifier"] = agree
    print(f"  agreement with exp06 pseudo-labels = {agree:.4f}  "
          f"(estimates Kaggle accuracy of this submission)")

if args.out:
    # Kaggle scorer expects "id", not "sample_index" as in the cached sample submission
    sub = pd.DataFrame({"id": meta_te["sample_index"], "is_forget": preds_te})
    sub.to_csv(args.out, index=False)
    print(f"  wrote {args.out}")

out_json = Path(args.ckpt).parent / "robust_eval.json"
out_json.write_text(json.dumps(results, indent=2))
print(f"Saved -> {out_json}")
