"""Evaluate the pretrained checkpoint via the full MIA pipeline.

Pipeline (mirrors competition_notebook.ipynb):
  1. CNN predicts (x,y) on all train samples.
  2. Compute Euclidean prediction errors.
  3. Train LogisticRegression(error -> is_forget) on train data.
  4. Evaluate LR on train (self-eval) — the notebook's reported accuracy.
  5. Optionally compute retain/forget localization stats separately.

Saves results to experiments/exp00_baseline/metrics.json.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import load_model
from src.dataset import format_csi_for_cnn
from src.metrics import get_predictions, prediction_errors, localization_stats, mia_accuracy

DATA = ROOT / "data" / "public"
CKPT = ROOT / "data" / "baseline_cnn_task2.pth"
OUT_DIR = ROOT / "experiments" / "exp00_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

model = load_model(str(CKPT), DEVICE)
model.eval()
print(f"Loaded {CKPT.name}\n")

# ---- Load & preprocess train ----
print("Loading train data...")
csi_train = np.load(DATA / "task2_train_csi.npy")
pos_train = np.load(DATA / "task2_train_positions.npy")
meta_train = pd.read_csv(DATA / "task2_train_metadata.csv")

X_train = format_csi_for_cnn(csi_train)
Y_train = pos_train[:, :2]
labels = meta_train["is_forget"].values

print(f"  Train: {X_train.shape}  retain={( labels==0).sum()}  forget={(labels==1).sum()}")

# ---- CNN predictions on train ----
print("Running CNN on train set...")
preds_train = get_predictions(model, X_train, device=DEVICE)
errors_train = prediction_errors(preds_train, Y_train)

# ---- MIA: LR on train errors, evaluate on train (self-eval like notebook) ----
mia = mia_accuracy(errors_train, labels, errors_train, labels)
print(f"\nMIA train self-accuracy: {mia['mia_accuracy']*100:.2f}%  "
      f"(notebook baseline: ~82.84%)")

# ---- Per-split localization stats ----
retain_mask = labels == 0
forget_mask = labels == 1

results = {
    "mia_train_self_accuracy": mia["mia_accuracy"],
    "retain": localization_stats(errors_train[retain_mask]),
    "forget": localization_stats(errors_train[forget_mask]),
    "all_train": localization_stats(errors_train),
}

print("\nLocalization errors (2D Euclidean, metres):")
for split in ("retain", "forget", "all_train"):
    s = results[split]
    print(f"  {split:10s}  n={s['n_samples']:5d}  "
          f"mean={s['mean_m']:.4f}m  median={s['median_m']:.4f}m")

# ---- Test set (if positions available) ----
test_pos_path = ROOT / "data" / "task2_test_positions.npy"
test_csi_path = DATA / "task2_test_csi.npy"
if test_pos_path.exists() and test_csi_path.exists():
    print("\nLoading test data...")
    csi_test = np.load(test_csi_path)
    pos_test = np.load(test_pos_path)
    X_test = format_csi_for_cnn(csi_test)
    Y_test = pos_test[:, :2]
    preds_test = get_predictions(model, X_test, device=DEVICE)
    errors_test = prediction_errors(preds_test, Y_test)
    results["test"] = localization_stats(errors_test)
    s = results["test"]
    print(f"  {'test':10s}  n={s['n_samples']:5d}  "
          f"mean={s['mean_m']:.4f}m  median={s['median_m']:.4f}m")

(OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved -> {OUT_DIR / 'metrics.json'}")
