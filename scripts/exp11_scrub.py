"""Experiment 11 — SCRUB unlearning (cross-fitted, rules-safe).

SCRUB for regression: alternating epochs designed for the 'Resolving Confusion'
(mislabeled-data) scenario where unlearning = remove corrupted-label influence.

  max-step: gradient ASCENT on forget position loss (corrupted labels)
            → forget errors increase
  min-step: gradient DESCENT on retain position loss (clean labels)
            → retain errors stay low

Alternating avoids the entanglement that killed NegGrad+/SSD: after each max-step
the model has a chance to repair collateral retain damage before the next ascent.

Starting point: exp07 fold checkpoints (already relabeled + finetuned).
SCRUB adds further forget amplification while anchoring retain quality.
OOF error cross-fitting avoids the memorisation bias that makes train-LR thresholds
fail to transfer to test.
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
from src.metrics import (get_predictions, prediction_errors, mia_accuracy,
                         gmm_threshold_predictions, localization_stats)

CFG = {
    # SCRUB schedule
    "n_cycles": 15,             # max-step + min-step cycles per fold
    "max_steps_per_cycle": 1,   # ascent epochs on forget set per cycle
    "min_steps_per_cycle": 2,   # descent epochs on retain set per cycle
    "lr_max": 2e-5,             # ascent learning rate (small: controlled ascent)
    "lr_min": 5e-5,             # descent learning rate
    "max_grad_norm": 1.0,       # clip gradients in max-step
    "retain_err_ceiling": 0.25, # early-stop cycle if retain err exceeds this (m)
    # data
    "batch_size": 64,
    "n_folds": 5,
    "seed": 42,
    "k_neighbours": 10,
    # starting checkpoints
    "start_from_exp07": True,   # True = load exp07 fold ckpts; False = baseline CNN
}

DATA = ROOT / "data" / "public"
CKPT_BASELINE = ROOT / "data" / "baseline_cnn_task2.pth"
EXP07_DIR = ROOT / "experiments" / "exp07_crossfit"
OUT_DIR = ROOT / "experiments" / "exp11_scrub"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PSEUDO_PATH = ROOT / "experiments" / "exp06_direct_classifier" / "test_proba.npy"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(CFG["seed"])
print(f"Device: {DEVICE}")

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
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

X_all = format_csi_for_cnn(csi_train)
X_test = format_csi_for_cnn(csi_test)
n = len(y)

# --------------------------------------------------------------------------- #
# kNN-corrected labels (exp05 recipe; same across all folds)
# --------------------------------------------------------------------------- #
print("Correcting forget labels via retain CSI-neighbours...")
mag = np.abs(csi_train.reshape(n, -1)).astype(np.float32)
pca = PCA(n_components=64, random_state=CFG["seed"])
feats = pca.fit_transform(mag)
nn_idx = (NearestNeighbors(n_neighbors=CFG["k_neighbours"])
          .fit(feats[retain_mask])
          .kneighbors(feats[forget_mask], return_distance=False))
pos_corrected = pos_train.copy()
pos_corrected[forget_mask] = pos_train[retain_mask][nn_idx].mean(axis=1)
Y_corrected = torch.tensor(pos_corrected, dtype=torch.float32)

# --------------------------------------------------------------------------- #
# SCRUB helpers
# --------------------------------------------------------------------------- #
criterion = nn.MSELoss()


def scrub_fold(model: nn.Module,
               X_tr_forget: torch.Tensor, Y_tr_forget_corrupted: torch.Tensor,
               X_tr_retain: torch.Tensor, Y_tr_retain: torch.Tensor,
               pos_tr_forget_orig: np.ndarray,
               pos_tr_retain_orig: np.ndarray) -> list[dict]:
    """Apply SCRUB to one fold model in-place. Returns per-cycle history."""
    opt_max = torch.optim.SGD(model.parameters(), lr=CFG["lr_max"])
    opt_min = torch.optim.Adam(model.parameters(), lr=CFG["lr_min"])

    forget_loader = DataLoader(
        TensorDataset(X_tr_forget, Y_tr_forget_corrupted),
        batch_size=CFG["batch_size"], shuffle=True, num_workers=0)
    retain_loader = DataLoader(
        TensorDataset(X_tr_retain, Y_tr_retain),
        batch_size=CFG["batch_size"], shuffle=True, num_workers=0)

    history = []
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_mia = 0.0

    for cycle in range(1, CFG["n_cycles"] + 1):
        # -- max-step(s): gradient ASCENT on corrupted forget labels --
        model.train()
        for _ in range(CFG["max_steps_per_cycle"]):
            for X_b, Y_b in forget_loader:
                X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
                opt_max.zero_grad()
                loss = criterion(model(X_b), Y_b)
                (-loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
                opt_max.step()

        # -- min-step(s): gradient DESCENT on clean retain labels --
        for _ in range(CFG["min_steps_per_cycle"]):
            for X_b, Y_b in retain_loader:
                X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
                opt_min.zero_grad()
                criterion(model(X_b), Y_b).backward()
                opt_min.step()

        # -- per-cycle diagnostics --
        model.eval()
        preds_f = get_predictions(model, X_tr_forget, device=DEVICE)
        preds_r = get_predictions(model, X_tr_retain, device=DEVICE)
        err_f = prediction_errors(preds_f, pos_tr_forget_orig).mean()
        err_r = prediction_errors(preds_r, pos_tr_retain_orig).mean()

        # crude self-MIA on training split
        nf, nr = len(pos_tr_forget_orig), len(pos_tr_retain_orig)
        all_errs = np.concatenate([
            prediction_errors(preds_f, pos_tr_forget_orig),
            prediction_errors(preds_r, pos_tr_retain_orig),
        ])
        all_labels = np.array([1] * nf + [0] * nr)
        mia_r = mia_accuracy(all_errs, all_labels, all_errs, all_labels)
        mia_acc = mia_r["mia_accuracy"]

        rec = {"cycle": cycle,
               "forget_err": float(err_f),
               "retain_err": float(err_r),
               "mia_self": float(mia_acc)}
        history.append(rec)

        if mia_acc > best_mia:
            best_mia = mia_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if err_r > CFG["retain_err_ceiling"]:
            print(f"      cycle {cycle}: early stop (retain {err_r:.3f}m > ceiling)")
            break

    model.load_state_dict(best_state)
    return history


# --------------------------------------------------------------------------- #
# Cross-fitting
# --------------------------------------------------------------------------- #
oof_errors = np.zeros(n)
test_errors_per_fold = []
skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])

for fold, (tr_idx, ho_idx) in enumerate(skf.split(np.zeros(n), y), start=1):
    fold_ckpt = OUT_DIR / f"model_fold{fold}.pth"
    fold_oof = OUT_DIR / f"oof_errors_fold{fold}.npy"
    fold_test = OUT_DIR / f"test_errors_fold{fold}.npy"
    fold_hist = OUT_DIR / f"history_fold{fold}.json"

    if fold_oof.exists() and fold_test.exists():
        print(f"\n[fold {fold}/{CFG['n_folds']}]  cached — skipping")
        oof_errors[ho_idx] = np.load(fold_oof)
        test_errors_per_fold.append(np.load(fold_test))
        continue

    print(f"\n[fold {fold}/{CFG['n_folds']}]  train={len(tr_idx)}  holdout={len(ho_idx)}")

    # Load starting checkpoint
    if CFG["start_from_exp07"]:
        src = EXP07_DIR / f"model_fold{fold}.pth"
        if not src.exists():
            print(f"  exp07 fold{fold} not found, falling back to baseline")
            src = CKPT_BASELINE
    else:
        src = CKPT_BASELINE

    model = load_model(str(src), DEVICE)

    if not fold_ckpt.exists():
        # Split training indices into forget/retain
        tr_forget_idx = tr_idx[y[tr_idx] == 1]
        tr_retain_idx = tr_idx[y[tr_idx] == 0]

        X_tr_forget = X_all[tr_forget_idx]
        Y_tr_forget_corrupted = torch.tensor(pos_train[tr_forget_idx], dtype=torch.float32)
        X_tr_retain = X_all[tr_retain_idx]
        Y_tr_retain = torch.tensor(pos_train[tr_retain_idx], dtype=torch.float32)

        print(f"  SCRUB: {len(tr_forget_idx)} forget  {len(tr_retain_idx)} retain  "
              f"→ {CFG['n_cycles']} cycles")

        hist = scrub_fold(
            model,
            X_tr_forget, Y_tr_forget_corrupted,
            X_tr_retain, Y_tr_retain,
            pos_train[tr_forget_idx],
            pos_train[tr_retain_idx],
        )
        final = hist[-1]
        print(f"  final cycle: forget={final['forget_err']:.4f}m  "
              f"retain={final['retain_err']:.4f}m  MIA={final['mia_self']:.4f}")

        torch.save(model.state_dict(), fold_ckpt)
        fold_hist.write_text(json.dumps(hist, indent=2))
    else:
        print("  checkpoint found — recomputing errors")
        model = load_model(str(fold_ckpt), DEVICE)

    # Errors vs ORIGINAL (corrupted) labels — what the Kaggle detector sees
    preds_ho = get_predictions(model, X_all[ho_idx], device=DEVICE)
    fold_oof_errors = prediction_errors(preds_ho, pos_train[ho_idx])
    oof_errors[ho_idx] = fold_oof_errors

    preds_te = get_predictions(model, X_test, device=DEVICE)
    fold_test_errors = prediction_errors(preds_te, pos_test)
    test_errors_per_fold.append(fold_test_errors)

    np.save(fold_oof, fold_oof_errors)
    np.save(fold_test, fold_test_errors)
    print(f"  holdout: forget mean={fold_oof_errors[y[ho_idx]==1].mean():.4f}m  "
          f"retain mean={fold_oof_errors[y[ho_idx]==0].mean():.4f}m")

test_errors = np.mean(test_errors_per_fold, axis=0)
np.save(OUT_DIR / "oof_errors.npy", oof_errors)
np.save(OUT_DIR / "test_errors.npy", test_errors)

# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
print("\n=== Detectors ===")
results = {"config": CFG}
results["oof_retain"] = localization_stats(oof_errors[retain_mask])
results["oof_forget"] = localization_stats(oof_errors[forget_mask])
print(f"OOF errors: retain mean={oof_errors[retain_mask].mean():.4f}m  "
      f"forget mean={oof_errors[forget_mask].mean():.4f}m")

mia = mia_accuracy(oof_errors, y, oof_errors, y)
lr_preds_test = mia["lr_model"].predict(test_errors.reshape(-1, 1)).astype(int)
results["oof_lr_train_accuracy"] = mia["mia_accuracy"]
results["lr_test_forget_rate"] = float(lr_preds_test.mean())
print(f"LR on OOF errors: train acc={mia['mia_accuracy']:.4f}  "
      f"test forget rate={lr_preds_test.mean():.4f}")

gmm = gmm_threshold_predictions(test_errors)
gmm_preds_test = gmm["predictions"]
results["gmm_test_forget_rate"] = gmm["forget_rate"]
print(f"GMM on test errors: forget rate={gmm['forget_rate']:.4f}  "
      f"component means={[f'{m:.3f}' for m in gmm['component_means_m']]} m")

results["lr_gmm_agreement"] = float((lr_preds_test == gmm_preds_test).mean())
print(f"LR vs GMM agreement on test: {results['lr_gmm_agreement']:.4f}")

if PSEUDO_PATH.exists():
    pseudo = (np.load(PSEUDO_PATH) > 0.5).astype(int)
    results["diag_lr_pseudo_agreement"] = float((lr_preds_test == pseudo).mean())
    results["diag_gmm_pseudo_agreement"] = float((gmm_preds_test == pseudo).mean())
    print(f"[diag] pseudo-label agreement: LR={results['diag_lr_pseudo_agreement']:.4f}  "
          f"GMM={results['diag_gmm_pseudo_agreement']:.4f}")

for tag, preds in [("lr", lr_preds_test), ("gmm", gmm_preds_test)]:
    sub = pd.DataFrame({"id": meta_test["sample_index"], "is_forget": preds})
    path = OUT_DIR / f"submission_{tag}.csv"
    sub.to_csv(path, index=False)
    print(f"wrote {path}")

(OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved -> {OUT_DIR / 'metrics.json'}")
