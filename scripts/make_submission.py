"""Generate a Kaggle submission CSV from a trained checkpoint.

Usage:
  python scripts/make_submission.py [--ckpt path/to/model.pth] [--out submission.csv]

Pipeline:
  1. Load checkpoint.
  2. CNN predictions on full train set -> compute errors.
  3. Train LR(errors -> is_forget) on train.
  4. CNN predictions on test set -> compute errors (using test positions if available).
  5. LR predicts is_forget for test -> save CSV.
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
from src.metrics import get_predictions, prediction_errors, mia_accuracy

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=str(ROOT / "data" / "baseline_cnn_task2.pth"))
parser.add_argument("--out", default="submission.csv")
args = parser.parse_args()

DATA = ROOT / "data" / "public"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}  |  Checkpoint: {args.ckpt}")

model = load_model(args.ckpt, DEVICE)

# ---- Train set (to fit LR) ----
csi_train = np.load(DATA / "task2_train_csi.npy")
pos_train = np.load(DATA / "task2_train_positions.npy")
meta_train = pd.read_csv(DATA / "task2_train_metadata.csv")

X_train = format_csi_for_cnn(csi_train)
Y_train = pos_train[:, :2]
labels_train = meta_train["is_forget"].values

print("Running CNN on train set...")
preds_train = get_predictions(model, X_train, device=DEVICE)
errors_train = prediction_errors(preds_train, Y_train)

# ---- Test set ----
csi_test = np.load(DATA / "task2_test_csi.npy")
X_test = format_csi_for_cnn(csi_test)

test_pos_path = ROOT / "data" / "task2_test_positions.npy"
print("Running CNN on test set...")
preds_test = get_predictions(model, X_test, device=DEVICE)

if test_pos_path.exists():
    Y_test = np.load(test_pos_path)[:, :2]
    errors_test = prediction_errors(preds_test, Y_test)
    print("  Using true test positions for error computation.")
else:
    errors_test = np.linalg.norm(preds_test, axis=1)
    print("  WARNING: test positions not found — using prediction magnitude as proxy.")

# ---- Fit LR on train, predict on test ----
mia = mia_accuracy(errors_train, labels_train, errors_test, np.zeros(len(errors_test)))
lr = mia["lr_model"]
test_preds = lr.predict(errors_test.reshape(-1, 1))

# ---- Build submission ----
sample_sub = pd.read_csv(DATA / "task2_sample_submission.csv")
sub = sample_sub.copy().rename(columns={"sample_index": "id"})
sub["is_forget"] = test_preds
sub = sub[["id", "is_forget"]]
sub.to_csv(args.out, index=False)
print(f"\nSubmission saved to {args.out}")
print(sub.head())
print(f"is_forget distribution:\n{sub['is_forget'].value_counts().to_string()}")
