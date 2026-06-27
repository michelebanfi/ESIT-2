"""Activation-space diagnostic probes for the baseline checkpoint.

--probe    Per-layer separability: extract channel-mean post-ReLU activations at each
           of the 6 blocks, compute Cohen's d per channel and a 5-fold linear probe AUC.
           Also computes the diagonal Fisher importance (I_forget vs I_retain) to
           characterise which parameter blocks carry the forget influence.

--ssd      Selective Synaptic Dampening (negative result): dampen parameters where
           I_forget >> I_retain, grid over (alpha, lambda), BN recalibrate on retain.
           Result: forget/retain circuits are entangled — nothing in the grid passes the
           retain-error guard. Provided for completeness.

Artifacts written to experiments/probe/:
  cohens_d_layer{2,6,10,14,18,21}.npy     per-channel Cohen's d vectors
  layer_separability.json                  probe AUC + Cohen's d stats per block
  fisher_layer_summary.json                forget/retain Fisher ratio stats per tensor
  fisher_{forget,retain}_baseline_cnn_task2.pth   cached Fisher dicts
  ssd_grid_results_baseline.json           SSD grid (always best=null)

Usage:
  python scripts/probing/activations.py --probe
  python scripts/probing/activations.py --ssd
  python scripts/probing/activations.py --probe --ssd
"""
import sys
import json
import copy
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from src.model import load_model
from src.dataset import format_csi_for_cnn
from src.metrics import (get_predictions, prediction_errors, mia_accuracy,
                          gmm_threshold_predictions)
from src.activations import ActivationTap, fisher_importance

CFG = {
    "seed": 42,
    "batch_size": 256,
    # --- probe ---
    "probe_relu_idx": [2, 6, 10, 14, 18, 21],    # post-ReLU outputs, blocks 1-6
    "fisher_batch_size": 64,
    # --- ssd ---
    "ssd_alphas": [1.0, 2.0, 5.0, 10.0],
    "ssd_lambdas": [0.1, 0.5, 1.0],
    "ssd_exclude_bn": True,
    "retain_err_guard_m": 0.25,
    "gmm_rate_range": [0.40, 0.60],
}

DATA = ROOT / "data" / "public"
OUT_DIR = ROOT / "experiments" / "probe"
EPS = 1e-8

parser = argparse.ArgumentParser()
parser.add_argument("--probe", action="store_true")
parser.add_argument("--ssd", action="store_true")
parser.add_argument("--ckpt-in", default=str(ROOT / "data" / "baseline_cnn_task2.pth"))
args = parser.parse_args()
if not (args.probe or args.ssd):
    parser.error("pick at least one of --probe / --ssd")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CKPT_STEM = Path(args.ckpt_in).stem
print(f"Device: {DEVICE}   ckpt-in: {args.ckpt_in}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")
csi_train = np.load(DATA / "task2_train_csi.npy")
pos_train = np.load(DATA / "task2_train_positions.npy")[:, :2].copy()
meta_train = pd.read_csv(DATA / "task2_train_metadata.csv")
y = meta_train["is_forget"].values
forget_mask = y == 1
retain_mask = ~forget_mask

csi_test = np.load(DATA / "task2_test_csi.npy")
pos_test = np.load(ROOT / "data" / "task2_test_positions.npy")[:, :2]

X_full = format_csi_for_cnn(csi_train)
X_test = format_csi_for_cnn(csi_test)


# ---------------------------------------------------------------- shared helpers

def quick_eval(model):
    """Self-MIA + retain/forget errors on train; official-LR forget rate on test."""
    err_tr = prediction_errors(get_predictions(model, X_full, device=DEVICE), pos_train)
    mia = mia_accuracy(err_tr, y, err_tr, y)
    row = {
        "self_mia": mia["mia_accuracy"],
        "retain_err_m": float(err_tr[retain_mask].mean()),
        "forget_err_m": float(err_tr[forget_mask].mean()),
    }
    err_te = prediction_errors(get_predictions(model, X_test, device=DEVICE), pos_test)
    lr_te = mia["lr_model"].predict(err_te.reshape(-1, 1)).astype(int)
    row["lr_test_forget_rate"] = float(lr_te.mean())
    gmm = gmm_threshold_predictions(err_te)
    row["gmm_test_forget_rate"] = gmm["forget_rate"]
    return row


def get_fisher(model):
    """Load cached Fisher dicts or compute them."""
    f_path = OUT_DIR / f"fisher_forget_{CKPT_STEM}.pth"
    r_path = OUT_DIR / f"fisher_retain_{CKPT_STEM}.pth"
    if f_path.exists() and r_path.exists():
        print("Loading cached Fisher importances...")
        return torch.load(f_path, weights_only=True), torch.load(r_path, weights_only=True)
    Y_t = torch.tensor(pos_train, dtype=torch.float32)
    print("Computing Fisher (forget)...")
    imp_f = fisher_importance(model, X_full[forget_mask], Y_t[forget_mask], DEVICE,
                               CFG["fisher_batch_size"])
    print("Computing Fisher (retain)...")
    imp_r = fisher_importance(model, X_full[retain_mask], Y_t[retain_mask], DEVICE,
                               CFG["fisher_batch_size"])
    torch.save(imp_f, f_path)
    torch.save(imp_r, r_path)
    return imp_f, imp_r


def bn_param_names(model):
    names = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            names.add(f"{mod_name}.weight")
            names.add(f"{mod_name}.bias")
    return names


def recalibrate_bn(model):
    """Reset BN running stats and re-estimate them on retain data only."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average
    model.eval()
    for m in bns:
        m.train()
    X_ret = X_full[retain_mask]
    with torch.no_grad():
        for i in range(0, len(X_ret), 256):
            model(X_ret[i:i + 256].to(DEVICE))
    model.eval()
    for m in bns:
        m.momentum = 0.1


# ---------------------------------------------------------------- --probe

def run_probe():
    print("\n=== PROBE: activation separability + Fisher importance ===")
    model = load_model(args.ckpt_in, DEVICE)
    model.eval()

    idx = CFG["probe_relu_idx"]
    tap = ActivationTap(model, idx)
    feats = {i: [] for i in idx}
    try:
        with torch.no_grad():
            for s in range(0, len(X_full), CFG["batch_size"]):
                model(X_full[s:s + CFG["batch_size"]].to(DEVICE))
                for i in idx:
                    feats[i].append(tap.acts[i].mean(dim=(2, 3)).cpu().numpy())
                tap.clear()
    finally:
        tap.remove()
    feats = {i: np.concatenate(v) for i, v in feats.items()}

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    sep = {}
    cv = StratifiedKFold(5, shuffle=True, random_state=CFG["seed"])
    for i in idx:
        Fm = feats[i]
        mu_f, mu_r = Fm[forget_mask].mean(0), Fm[retain_mask].mean(0)
        s_f, s_r = Fm[forget_mask].std(0), Fm[retain_mask].std(0)
        n_f, n_r = forget_mask.sum(), retain_mask.sum()
        pooled = np.sqrt(((n_f - 1) * s_f ** 2 + (n_r - 1) * s_r ** 2) / (n_f + n_r - 2))
        d = (mu_f - mu_r) / (pooled + EPS)
        np.save(OUT_DIR / f"cohens_d_layer{i}.npy", d)
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        auc = float(cross_val_score(probe, Fm, y, cv=cv, scoring="roc_auc").mean())
        sep[str(i)] = {
            "n_channels": int(Fm.shape[1]),
            "probe_auc_cv": auc,
            "mean_abs_d": float(np.abs(d).mean()),
            "max_abs_d": float(np.abs(d).max()),
            "n_channels_absd_gt_0.5": int((np.abs(d) > 0.5).sum()),
        }
        print(f"  layer {i:2d} (C={Fm.shape[1]:3d}):  probe AUC={auc:.4f}  "
              f"mean|d|={sep[str(i)]['mean_abs_d']:.3f}  "
              f"#|d|>0.5={sep[str(i)]['n_channels_absd_gt_0.5']}")
    (OUT_DIR / "layer_separability.json").write_text(json.dumps(sep, indent=2))

    # Fisher importance
    imp_f, imp_r = get_fisher(model)
    summary = {}
    for n in imp_f:
        ratio = (imp_f[n] / (imp_r[n] + 1e-12)).flatten()
        summary[n] = {
            "numel": int(ratio.numel()),
            "frac_ratio_gt_2": float((ratio > 2).float().mean()),
            "frac_ratio_gt_5": float((ratio > 5).float().mean()),
            "frac_ratio_gt_10": float((ratio > 10).float().mean()),
        }
    (OUT_DIR / "fisher_layer_summary.json").write_text(json.dumps(summary, indent=2))
    print("  Fisher ratio I_f/I_r > 5, per parameter tensor:")
    for n, s in summary.items():
        if s["frac_ratio_gt_5"] > 0.01:
            print(f"    {n:35s} frac>5 = {s['frac_ratio_gt_5']:.3f}")


# ---------------------------------------------------------------- --ssd

def run_ssd():
    print("\n=== SSD: selective synaptic dampening (negative result) ===")
    model = load_model(args.ckpt_in, DEVICE)
    imp_f, imp_r = get_fisher(model)
    base_sd = copy.deepcopy(model.state_dict())
    skip = bn_param_names(model) if CFG["ssd_exclude_bn"] else set()

    rows = []
    for alpha in CFG["ssd_alphas"]:
        for lam in CFG["ssd_lambdas"]:
            sd = copy.deepcopy(base_sd)
            n_damp = 0
            for n in sd:
                if n in skip or n not in imp_f:
                    continue
                If, Ir = imp_f[n].to(DEVICE), imp_r[n].to(DEVICE)
                sel = If > alpha * Ir
                if sel.any():
                    damp = torch.clamp(lam * Ir / (If + 1e-12), max=1.0)
                    sd[n][sel] *= damp[sel]
                    n_damp += int(sel.sum())
            model.load_state_dict(sd)
            recalibrate_bn(model)
            row = {"alpha": alpha, "lambda": lam, "n_dampened": n_damp}
            row.update(quick_eval(model))
            rows.append(row)
            print(f"  a={alpha:5.1f} l={lam:4.1f}  damp={n_damp:7d}  "
                  f"selfMIA={row['self_mia']:.4f}  retain={row['retain_err_m']:.4f}m  "
                  f"forget={row['forget_err_m']:.4f}m  gmm_rate={row['gmm_test_forget_rate']:.3f}")

    lo, hi = CFG["gmm_rate_range"]
    ok = [r for r in rows
          if r["retain_err_m"] <= CFG["retain_err_guard_m"] and lo <= r["gmm_test_forget_rate"] <= hi]
    best = max(ok, key=lambda r: r["self_mia"]) if ok else None

    out = {"ckpt_in": args.ckpt_in, "grid": rows, "best": best}
    (OUT_DIR / f"ssd_grid_results_{CKPT_STEM}.json").write_text(json.dumps(out, indent=2))
    if best is None:
        print("  !! no grid point passed the guards — nothing saved  (expected: entangled circuits)")
    else:
        print(f"  best: alpha={best['alpha']} lambda={best['lambda']}")


if __name__ == "__main__":
    if args.probe:
        run_probe()
    if args.ssd:
        run_ssd()
