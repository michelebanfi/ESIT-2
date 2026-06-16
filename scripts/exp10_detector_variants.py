"""Experiment 10 — improved detectors on top of existing checkpoints.

Two variants, no new training:

  --top50   Force the known test forget rate (≈0.50) as a hard prior: rank test samples
            by error, label the top-1364 as forget. Uses exp07 test errors.
            Eliminates all threshold calibration — no LR fit, no GMM component finding.

  --ensemble  Average exp07 and exp08-diverge test error vectors before running
            GMM + LR detection. The two models use orthogonal mechanisms (label-denoising
            vs activation-divergence); combining their errors may sharpen the two-component
            gap. OOF errors stay exp07 (the only cross-fitted set; diverge has no OOF).

Offline scoring: GMM forget rate (want ≈0.50) + agreement with exp06 pseudo-labels.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import load_model
from src.dataset import format_csi_for_cnn
from src.metrics import (get_predictions, prediction_errors,
                         mia_accuracy, gmm_threshold_predictions)

DATA = ROOT / "data" / "public"
EXP07_DIR = ROOT / "experiments" / "exp07_crossfit"
EXP08_DIV_DIR = ROOT / "experiments" / "exp08_activation_unlearning" / "diverge"
OUT_DIR = ROOT / "experiments" / "exp10_detector_variants"
PSEUDO_PATH = ROOT / "experiments" / "exp06_direct_classifier" / "test_proba.npy"

parser = argparse.ArgumentParser()
parser.add_argument("--top50", action="store_true")
parser.add_argument("--ensemble", action="store_true")
args = parser.parse_args()
if not (args.top50 or args.ensemble):
    parser.error("pick at least one of --top50 / --ensemble")

OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Loading data...")
meta_tr = pd.read_csv(DATA / "task2_train_metadata.csv")
meta_te = pd.read_csv(DATA / "task2_test_metadata.csv")
y_tr = meta_tr["is_forget"].values
PSEUDO = (np.load(PSEUDO_PATH) > 0.5).astype(int) if PSEUDO_PATH.exists() else None

# exp07 precomputed errors (already on disk)
oof07 = np.load(EXP07_DIR / "oof_errors.npy")
test07 = np.load(EXP07_DIR / "test_errors.npy")
N_TEST = len(test07)
N_FORGET_50 = N_TEST // 2   # 1364

print(f"exp07: oof retain={oof07[y_tr==0].mean():.4f}m  forget={oof07[y_tr==1].mean():.4f}m")
print(f"exp07: test mean={test07.mean():.4f}m  n={N_TEST}  50%={N_FORGET_50}")


def diag(tag, preds_te, test_errors=None):
    """Print + return diagnostic dict for a set of test predictions."""
    rate = float(preds_te.mean())
    row = {"tag": tag, "forget_rate": rate}
    if PSEUDO is not None:
        agree = float((preds_te == PSEUDO).mean())
        row["pseudo_agreement"] = agree
        print(f"  [{tag}]  forget_rate={rate:.4f}  pseudo_agree={agree:.4f}")
    else:
        print(f"  [{tag}]  forget_rate={rate:.4f}")
    if test_errors is not None:
        gmm = gmm_threshold_predictions(test_errors)
        row["gmm_forget_rate"] = gmm["forget_rate"]
        row["gmm_component_means_m"] = gmm["component_means_m"]
    return row


def write_submission(tag, preds_te):
    sub = pd.DataFrame({"id": meta_te["sample_index"], "is_forget": preds_te.astype(int)})
    path = OUT_DIR / f"submission_{tag}.csv"
    sub.to_csv(path, index=False)
    print(f"  wrote {path}")
    return path


results = {}

# ---------------------------------------------------------------- --top50
if args.top50:
    print("\n=== TOP-50: rank test errors, force 50% forget ===")
    # sort descending; top-1364 = forget
    rank = np.argsort(test07)[::-1]
    preds = np.zeros(N_TEST, dtype=int)
    preds[rank[:N_FORGET_50]] = 1

    mia = mia_accuracy(oof07, y_tr, oof07, y_tr)
    print(f"  exp07 OOF self-MIA (ref): {mia['mia_accuracy']:.4f}")
    print(f"  forced forget count: {preds.sum()} / {N_TEST}")

    row = diag("top50", preds, test07)
    row["oof_self_mia"] = mia["mia_accuracy"]
    write_submission("exp07_top50", preds)
    results["top50"] = row

# ---------------------------------------------------------------- --ensemble
if args.ensemble:
    print("\n=== ENSEMBLE: exp07 + exp08-diverge test errors ===")

    # compute exp08-diverge test errors (not pre-saved as a vector)
    div_ckpt = EXP08_DIV_DIR / "model_best_none.pth"
    print(f"  loading diverge checkpoint: {div_ckpt}")
    model_div = load_model(str(div_ckpt), DEVICE)
    model_div.eval()

    csi_te = np.load(DATA / "task2_test_csi.npy")
    pos_te = np.load(ROOT / "data" / "task2_test_positions.npy")[:, :2]
    X_te = format_csi_for_cnn(csi_te)

    preds_div = get_predictions(model_div, X_te, device=DEVICE)
    test_div = prediction_errors(preds_div, pos_te)
    np.save(OUT_DIR / "test_errors_div.npy", test_div)
    print(f"  diverge test errors: mean={test_div.mean():.4f}m  "
          f"min={test_div.min():.4f}  max={test_div.max():.4f}")

    # also need diverge train errors for OOF reference
    csi_tr = np.load(DATA / "task2_train_csi.npy")
    pos_tr = np.load(DATA / "task2_train_positions.npy")[:, :2]
    X_tr = format_csi_for_cnn(csi_tr)
    preds_div_tr = get_predictions(model_div, X_tr, device=DEVICE)
    train_div = prediction_errors(preds_div_tr, pos_tr)
    print(f"  diverge train: retain={train_div[y_tr==0].mean():.4f}m  "
          f"forget={train_div[y_tr==1].mean():.4f}m")

    # try three blending weights: equal, exp07-heavy, exp08-heavy
    weights = [("50_50", 0.5, 0.5), ("70_30", 0.7, 0.3), ("30_70", 0.3, 0.7)]
    best_agree, best_tag = -1, None
    ensemble_rows = []

    for wtag, w07, w08 in weights:
        blended_te = w07 * test07 + w08 * test_div
        blended_oof = w07 * oof07 + w08 * train_div   # approximate: div has no true OOF

        mia = mia_accuracy(blended_oof, y_tr, blended_oof, y_tr)
        gmm = gmm_threshold_predictions(blended_te)

        # LR predictions on test
        lr_preds = mia["lr_model"].predict(blended_te.reshape(-1, 1)).astype(int)
        # GMM predictions on test
        gmm_preds = gmm["predictions"]

        print(f"\n  [{wtag}]  w07={w07}  w08={w08}")
        print(f"    self-MIA (approx OOF): {mia['mia_accuracy']:.4f}")
        print(f"    gmm test rate: {gmm['forget_rate']:.4f}  "
              f"means={[round(m,3) for m in gmm['component_means_m']]}m")

        row_lr = diag(f"ens_{wtag}_lr", lr_preds, blended_te)
        row_gmm = diag(f"ens_{wtag}_gmm", gmm_preds)

        for rtag, rpreds, rrow in [("lr", lr_preds, row_lr), ("gmm", gmm_preds, row_gmm)]:
            write_submission(f"exp10_ens_{wtag}_{rtag}", rpreds)
            rrow["w07"] = w07
            rrow["w08"] = w08
            rrow["oof_self_mia"] = mia["mia_accuracy"]
            ensemble_rows.append(rrow)
            if rrow.get("pseudo_agreement", -1) > best_agree:
                best_agree = rrow["pseudo_agreement"]
                best_tag = rrow["tag"]

    print(f"\n  best blending: {best_tag}  pseudo_agree={best_agree:.4f}")
    results["ensemble"] = {"rows": ensemble_rows, "best": best_tag, "best_pseudo": best_agree}

# ---------------------------------------------------------------- save
path = OUT_DIR / "metrics.json"
merged = json.loads(path.read_text()) if path.exists() else {}
merged.update(results)
path.write_text(json.dumps(merged, indent=2))
print(f"\nSaved -> {path}")
