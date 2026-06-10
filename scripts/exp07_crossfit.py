"""Experiment 07 — cross-fitted exp05 ensemble. Pure CNN-error pipeline (rules-safe).

Why cross-fitting: a model finetuned on a sample memorises it, so its train error is
unrepresentatively small and an LR threshold fit on train errors does not transfer to
test. Here every train sample's error comes from a fold model that never saw it.

Recipe per fold (5 stratified folds over is_forget):
  1. Relabel forget samples with kNN-corrected positions (exp05 recipe, computed once).
  2. Finetune the baseline CNN on the corrected labels of the other 4 folds.
  3. Record errors (vs the ORIGINAL labels) on the held-out fold and on the test set.

Detector:
  - LR fit on out-of-fold train errors -> applied to fold-averaged test errors.
  - GMM split of fold-averaged test errors (unsupervised alternative).
Both submissions are written; pick via the diagnostics.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import load_model
from src.dataset import format_csi_for_cnn
from src.train import train_epoch
from src.metrics import (get_predictions, prediction_errors, mia_accuracy,
                         gmm_threshold_predictions, localization_stats)

CFG = {
    "lr": 1e-4,
    "epochs": 30,
    "batch_size": 64,
    "n_folds": 5,
    "seed": 42,
    "k_neighbours": 10,
}

DATA = ROOT / "data" / "public"
CKPT_IN = ROOT / "data" / "baseline_cnn_task2.pth"
OUT_DIR = ROOT / "experiments" / "exp07_crossfit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PSEUDO_PATH = ROOT / "experiments" / "exp06_direct_classifier" / "test_proba.npy"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(CFG["seed"])
print(f"Device: {DEVICE}")

print("Loading data...")
csi_train = np.load(DATA / "task2_train_csi.npy")
pos_train = np.load(DATA / "task2_train_positions.npy")[:, :2].copy()
meta_train = pd.read_csv(DATA / "task2_train_metadata.csv")
y = meta_train["is_forget"].values
forget_mask = y == 1
retain_mask = ~forget_mask

csi_test = np.load(DATA / "task2_test_csi.npy")
pos_test = np.load(ROOT / "data" / "task2_test_positions.npy")[:, :2]
meta_test = pd.read_csv(DATA / "task2_test_metadata.csv")

# ---- kNN-corrected labels (exp05 recipe, train-data only) ----
print("Correcting forget labels via retain CSI-neighbours...")
n = len(csi_train)
mag = np.abs(csi_train.reshape(n, -1)).astype(np.float32)
pca = PCA(n_components=64, random_state=CFG["seed"])
feats = pca.fit_transform(mag)
nn_idx = (NearestNeighbors(n_neighbors=CFG["k_neighbours"])
          .fit(feats[retain_mask])
          .kneighbors(feats[forget_mask], return_distance=False))
pos_corrected = pos_train.copy()
pos_corrected[forget_mask] = pos_train[retain_mask][nn_idx].mean(axis=1)

X_full = format_csi_for_cnn(csi_train)
Y_corrected = torch.tensor(pos_corrected, dtype=torch.float32)
X_test = format_csi_for_cnn(csi_test)

# ---- Cross-fitting ----
oof_errors = np.zeros(n)
test_errors_per_fold = []
skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])

for fold, (tr_idx, ho_idx) in enumerate(skf.split(np.zeros(n), y), start=1):
    fold_ckpt = OUT_DIR / f"model_fold{fold}.pth"
    fold_oof = OUT_DIR / f"oof_errors_fold{fold}.npy"
    fold_test = OUT_DIR / f"test_errors_fold{fold}.npy"

    # resume: skip folds whose error arrays are already on disk
    if fold_oof.exists() and fold_test.exists():
        print(f"\n[fold {fold}/{CFG['n_folds']}]  cached — skipping")
        oof_errors[ho_idx] = np.load(fold_oof)
        test_errors_per_fold.append(np.load(fold_test))
        continue

    print(f"\n[fold {fold}/{CFG['n_folds']}]  train={len(tr_idx)}  holdout={len(ho_idx)}")
    if fold_ckpt.exists():
        print("  checkpoint found — recomputing errors without retraining")
        model = load_model(str(fold_ckpt), DEVICE)
    else:
        model = load_model(str(CKPT_IN), DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
        criterion = nn.MSELoss()
        loader = DataLoader(TensorDataset(X_full[tr_idx], Y_corrected[tr_idx]),
                            batch_size=CFG["batch_size"], shuffle=True, num_workers=0)
        for epoch in range(1, CFG["epochs"] + 1):
            t_loss = train_epoch(model, loader, optimizer, criterion, DEVICE)
            if epoch % 10 == 0:
                print(f"  epoch {epoch:2d}  train_mse={t_loss:.5f}")
        torch.save(model.state_dict(), fold_ckpt)

    # errors vs ORIGINAL (corrupted) labels — what the detector sees
    preds_ho = get_predictions(model, X_full[ho_idx], device=DEVICE)
    fold_oof_errors = prediction_errors(preds_ho, pos_train[ho_idx])
    oof_errors[ho_idx] = fold_oof_errors
    preds_te = get_predictions(model, X_test, device=DEVICE)
    fold_test_errors = prediction_errors(preds_te, pos_test)
    test_errors_per_fold.append(fold_test_errors)
    np.save(fold_oof, fold_oof_errors)
    np.save(fold_test, fold_test_errors)

test_errors = np.mean(test_errors_per_fold, axis=0)
np.save(OUT_DIR / "oof_errors.npy", oof_errors)
np.save(OUT_DIR / "test_errors.npy", test_errors)

# ---- Detectors ----
print("\n=== Detectors ===")
results = {"config": CFG}
results["oof_retain"] = localization_stats(oof_errors[retain_mask])
results["oof_forget"] = localization_stats(oof_errors[forget_mask])
print(f"OOF errors: retain mean={oof_errors[retain_mask].mean():.4f}m  "
      f"forget mean={oof_errors[forget_mask].mean():.4f}m")

# LR on out-of-fold errors (its train accuracy is itself cross-validated)
mia = mia_accuracy(oof_errors, y, oof_errors, y)
lr_preds_test = mia["lr_model"].predict(test_errors.reshape(-1, 1)).astype(int)
results["oof_lr_train_accuracy"] = mia["mia_accuracy"]
results["lr_test_forget_rate"] = float(lr_preds_test.mean())
print(f"LR on OOF errors: train acc={mia['mia_accuracy']:.4f}  "
      f"test forget rate={lr_preds_test.mean():.4f}")

# GMM split of test errors
gmm = gmm_threshold_predictions(test_errors)
gmm_preds_test = gmm["predictions"]
results["gmm_test_forget_rate"] = gmm["forget_rate"]
print(f"GMM on test errors: forget rate={gmm['forget_rate']:.4f}  "
      f"component means={[f'{m:.3f}' for m in gmm['component_means_m']]} m")

results["lr_gmm_agreement"] = float((lr_preds_test == gmm_preds_test).mean())
print(f"LR vs GMM agreement on test: {results['lr_gmm_agreement']:.4f}")

# diagnostic only — NOT part of the submission pipeline
if PSEUDO_PATH.exists():
    pseudo = (np.load(PSEUDO_PATH) > 0.5).astype(int)
    results["diag_lr_pseudo_agreement"] = float((lr_preds_test == pseudo).mean())
    results["diag_gmm_pseudo_agreement"] = float((gmm_preds_test == pseudo).mean())
    print(f"[diagnostic] pseudo-label agreement: LR={results['diag_lr_pseudo_agreement']:.4f}  "
          f"GMM={results['diag_gmm_pseudo_agreement']:.4f}")

for tag, preds in [("lr", lr_preds_test), ("gmm", gmm_preds_test)]:
    # Kaggle scorer expects "id", not "sample_index" as in the cached sample submission
    sub = pd.DataFrame({"id": meta_test["sample_index"], "is_forget": preds})
    path = OUT_DIR / f"submission_{tag}.csv"
    sub.to_csv(path, index=False)
    print(f"wrote {path}")

(OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
print(f"Saved -> {OUT_DIR / 'metrics.json'}")
