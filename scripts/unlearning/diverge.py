"""Unlearning via activation-trajectory divergence (exp08, winning method).

The key insight from probing: the CNN already "knows" forget samples — post-ReLU
activations at block 5 are linearly separable (AUC 0.986).  Parameter-space dampening
fails because forget/retain Fisher importances are entangled.  Solution: act in
activation space, at the trajectory level.

Method (trajectory-divergence finetune):
  - Frozen teacher = baseline checkpoint (reference trajectory).
  - Retain anchor: penalise deviation of the student's post-ReLU activations from the
    teacher's, on retain samples  (retain stays on trajectory).
  - Forget diverge: hinge-cosine loss on pre-ReLU BatchNorm activations for forget
    samples  (forget pushed off trajectory; BN outputs are signed — cosine push is
    meaningful without a trivial all-zero solution; norm-keeper prevents collapse).
  - Optional kNN forget-position loss (--forget-target knn): supplies the exp05
    transferable error signal on top of the activation divergence.

Best run: --forget-target knn --beta-anchor 3 --epochs 30  (checkpoint ep24)
  Offline LR~acc = 0.8442, forget-rate = 0.519, retainE = 0.139 m, self-MIA = 0.881.

Checkpoint: experiments/diverge/model_official_knn_b3.pth
Evaluate:   python scripts/submission/eval_official.py --ckpt experiments/diverge/model_official_knn_b3.pth

Usage:
  python scripts/unlearning/diverge.py --forget-target knn --beta-anchor 3 --epochs 30
"""
import sys
import json
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
from src.dataset import format_csi_for_cnn, knn_corrected_positions
from src.metrics import (get_predictions, prediction_errors, mia_accuracy,
                          gmm_threshold_predictions)
from src.activations import ActivationTap

CFG = {
    "seed": 42,
    "batch_size": 64,
    # --- diverge ---
    "div_bn_idx": [13, 17, 20],       # pre-ReLU BN outputs, blocks 4-6
    "anchor_relu_idx": [14, 18, 21],  # post-ReLU outputs, blocks 4-6
    "lr": 5e-5,
    "epochs": 12,                     # override with --epochs
    "beta_anchor": 1.0,               # override with --beta-anchor (winning: 3.0)
    "gamma_diverge": 0.5,
    "gamma_warmup_epochs": 2,
    "cos_margin": 0.0,
    "norm_keep_weight": 0.1,
    "delta_forget_pos": 1.0,
    "k_neighbours": 10,
    "eval_every": 3,
    "retain_err_guard_m": 0.25,
}

DATA = ROOT / "data" / "public"
OUT_DIR = ROOT / "experiments" / "diverge"
PSEUDO_PATH = ROOT / "experiments" / "pseudo_labels" / "test_proba.npy"
EPS = 1e-8

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ckpt-in", default=str(ROOT / "data" / "baseline_cnn_task2.pth"))
parser.add_argument("--forget-target", choices=["none", "knn"], default="none",
                    help="knn = add kNN-corrected forget-position loss (recommended)")
parser.add_argument("--epochs", type=int, default=None,
                    help="override CFG['epochs']")
parser.add_argument("--beta-anchor", type=float, default=None,
                    help="retain-anchor strength β (winning: 3.0; larger = tighter retain "
                         "trajectory = smaller train retainE = LR threshold shifts)")
args = parser.parse_args()

if args.epochs is not None:
    CFG["epochs"] = args.epochs
if args.beta_anchor is not None:
    CFG["beta_anchor"] = args.beta_anchor

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(CFG["seed"])
print(f"Device: {DEVICE}   ckpt-in: {args.ckpt_in}")
print(f"Config: forget-target={args.forget_target}  epochs={CFG['epochs']}  "
      f"beta_anchor={CFG['beta_anchor']}")

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
PSEUDO = (np.load(PSEUDO_PATH) > 0.5).astype(int) if PSEUDO_PATH.exists() else None


# ---------------------------------------------------------------- helpers

def quick_eval(model, with_test: bool = True) -> dict:
    """Self-MIA + retain/forget errors on train; official-LR proxy on test."""
    err_tr = prediction_errors(get_predictions(model, X_full, device=DEVICE), pos_train)
    mia = mia_accuracy(err_tr, y, err_tr, y)
    row = {
        "self_mia": mia["mia_accuracy"],
        "retain_err_m": float(err_tr[retain_mask].mean()),
        "forget_err_m": float(err_tr[forget_mask].mean()),
    }
    if with_test:
        err_te = prediction_errors(get_predictions(model, X_test, device=DEVICE), pos_test)
        lr_te = mia["lr_model"].predict(err_te.reshape(-1, 1)).astype(int)
        row["lr_test_forget_rate"] = float(lr_te.mean())
        gmm = gmm_threshold_predictions(err_te)
        row["gmm_test_forget_rate"] = gmm["forget_rate"]
        if PSEUDO is not None:
            row["lr_pseudo_agreement"] = float((lr_te == PSEUDO).mean())
            row["gmm_pseudo_agreement"] = float((gmm["predictions"] == PSEUDO).mean())
    return row


def set_train_frozen_bn(model):
    """Train mode but BN running stats frozen (affine params still learn)."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


# ---------------------------------------------------------------- run_diverge

def run_diverge():
    variant = args.forget_target
    print(f"\n=== DIVERGE: trajectory-divergence finetune (forget-target={variant}) ===")
    student = load_model(args.ckpt_in, DEVICE)
    teacher = load_model(args.ckpt_in, DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    div_idx = CFG["div_bn_idx"]
    anc_idx = CFG["anchor_relu_idx"]
    all_idx = sorted(set(div_idx) | set(anc_idx))
    tap_s = ActivationTap(student, all_idx)
    tap_t = ActivationTap(teacher, all_idx)

    Y_orig = torch.tensor(pos_train, dtype=torch.float32)
    if variant == "knn":
        print("Computing kNN-corrected forget positions (exp05 recipe)...")
        Y_corr = torch.tensor(
            knn_corrected_positions(csi_train, pos_train, forget_mask,
                                    k=CFG["k_neighbours"], seed=CFG["seed"]),
            dtype=torch.float32,
        )
    else:
        Y_corr = Y_orig

    m_forget = torch.tensor(forget_mask)
    loader = DataLoader(
        TensorDataset(X_full, Y_orig, Y_corr, m_forget),
        batch_size=CFG["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(CFG["seed"]),
    )

    opt = torch.optim.Adam(student.parameters(), lr=CFG["lr"])
    sel_key = "lr_pseudo_agreement" if PSEUDO is not None else "self_mia"
    best = {sel_key: -1.0}
    run_tag = f"{variant}_b{CFG['beta_anchor']:g}"
    best_path = OUT_DIR / f"model_official_{run_tag}.pth"
    log = []

    try:
        for epoch in range(1, CFG["epochs"] + 1):
            set_train_frozen_bn(student)
            gamma_t = CFG["gamma_diverge"] * min(1.0, epoch / max(1, CFG["gamma_warmup_epochs"]))
            sums = {"pos_r": 0.0, "anc": 0.0, "div": 0.0, "norm": 0.0, "pos_f": 0.0}
            ratio_sum, ratio_n, nb = 0.0, 0, 0

            for Xb, yo, yc, mb in loader:
                Xb = Xb.to(DEVICE)
                yo = yo.to(DEVICE)
                yc = yc.to(DEVICE)
                mb = mb.to(DEVICE)
                r, f = ~mb, mb
                tap_s.clear()
                tap_t.clear()
                pred = student(Xb)
                with torch.no_grad():
                    teacher(Xb)

                loss = pred.sum() * 0.0

                # ---- retain: position MSE + anchor post-ReLU activations to teacher ----
                if r.any():
                    L_pos_r = F.mse_loss(pred[r], yo[r])
                    L_anc = pred.sum() * 0.0
                    for i in anc_idx:
                        a = tap_s.acts[i][r].flatten(1)
                        a0 = tap_t.acts[i][r].flatten(1)
                        L_anc = L_anc + (((a - a0) ** 2).sum(1)
                                         / (a0.pow(2).sum(1) + EPS)).mean()
                    L_anc = L_anc / len(anc_idx)
                    loss = loss + L_pos_r + CFG["beta_anchor"] * L_anc
                    sums["pos_r"] += L_pos_r.item()
                    sums["anc"] += L_anc.item()

                # ---- forget: hinge-cosine push on pre-ReLU BN + norm-keeper ----
                if f.any():
                    L_div = pred.sum() * 0.0
                    L_norm = pred.sum() * 0.0
                    for i in div_idx:
                        a = tap_s.acts[i][f].flatten(1)
                        a0 = tap_t.acts[i][f].flatten(1)
                        cos = F.cosine_similarity(a, a0, dim=1)
                        L_div = L_div + F.relu(cos - CFG["cos_margin"]).mean()
                        an, a0n = a.norm(dim=1), a0.norm(dim=1)
                        L_norm = L_norm + (((an - a0n) / (a0n + EPS)) ** 2).mean()
                        with torch.no_grad():
                            ratio_sum += (an / (a0n + EPS)).mean().item()
                            ratio_n += 1
                    L_div = L_div / len(div_idx)
                    L_norm = L_norm / len(div_idx)
                    loss = loss + gamma_t * L_div + CFG["norm_keep_weight"] * L_norm
                    sums["div"] += L_div.item()
                    sums["norm"] += L_norm.item()

                    # ---- optional: kNN-corrected forget position loss ----
                    if variant == "knn":
                        L_pos_f = F.mse_loss(pred[f], yc[f])
                        loss = loss + CFG["delta_forget_pos"] * L_pos_f
                        sums["pos_f"] += L_pos_f.item()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                nb += 1

            entry = {
                "epoch": epoch,
                "gamma_t": gamma_t,
                **{k: v / nb for k, v in sums.items()},
                "forget_norm_ratio": ratio_sum / max(1, ratio_n),
            }
            collapse = " !! norm ratio < 0.5 — collapse warning" \
                if entry["forget_norm_ratio"] < 0.5 else ""
            print(f"  ep {epoch:2d}  pos_r={entry['pos_r']:.4f}  anc={entry['anc']:.4f}  "
                  f"div={entry['div']:.4f}  norm={entry['norm']:.4f}  "
                  f"pos_f={entry['pos_f']:.4f}  |a|/|a0|={entry['forget_norm_ratio']:.3f}"
                  f"{collapse}")

            if epoch % CFG["eval_every"] == 0 or epoch == CFG["epochs"]:
                student.eval()
                ev = quick_eval(student, with_test=True)
                entry.update(ev)
                print(f"      selfMIA={ev['self_mia']:.4f}  retain={ev['retain_err_m']:.4f}m  "
                      f"forget={ev['forget_err_m']:.4f}m  "
                      f"LRfgt={ev.get('lr_test_forget_rate', float('nan')):.3f}  "
                      f"LR~acc={ev.get('lr_pseudo_agreement', float('nan')):.4f}")
                if (ev["retain_err_m"] <= CFG["retain_err_guard_m"]
                        and ev[sel_key] > best[sel_key]):
                    best = {**ev, "epoch": epoch}
                    torch.save(student.state_dict(), best_path)
                    print(f"      saved -> {best_path}  ({sel_key}={ev[sel_key]:.4f})")
            log.append(entry)
    finally:
        tap_s.remove()
        tap_t.remove()

    out = {
        "ckpt_in": args.ckpt_in,
        "variant": variant,
        "run_tag": run_tag,
        "select_on": sel_key,
        "config": CFG,
        "best": best,
        "log": log,
    }
    log_path = OUT_DIR / f"train_log_{run_tag}.json"
    log_path.write_text(json.dumps(out, indent=2))

    if best[sel_key] < 0:
        print("  !! no epoch passed the retain-error guard — nothing saved")
    else:
        print(f"  best epoch {best['epoch']} (by {sel_key}): "
              f"selfMIA={best['self_mia']:.4f}  "
              f"retain={best['retain_err_m']:.4f}m  forget={best['forget_err_m']:.4f}m  "
              f"LRfgt={best.get('lr_test_forget_rate', float('nan')):.3f}  "
              f"LR~acc={best.get('lr_pseudo_agreement', float('nan')):.4f}")
        print(f"  Evaluate: python scripts/submission/eval_official.py --ckpt {best_path}")


if __name__ == "__main__":
    run_diverge()
