"""Experiment 09 — concept-erasure unlearning ("Probe-Adversarial NullGrad").

Exploits the exp08 finding that forget samples are linearly separable in the CNN's
activation space (probe AUC 0.986 at block 5). Rather than pushing forget activations
*off-trajectory* (exp08 --diverge, which memorises train-forget-specific directions and
transfers poorly), we *erase the forget concept*: drive forget activations onto the
retain manifold along the direction a probe uses to detect them, so the network stops
treating contaminated samples specially and falls back to the generic CSI->position map.

Mechanism (why this lifts the Kaggle metric):
  The contamination is corrupted position labels; the baseline CNN partly *memorised*
  forget samples (CSI_forget -> corrupted_label), which is exactly the forget-specific
  representation the probe reads. Erasing it makes the model predict the *honest* (true)
  position for forget samples -- but their label is still corrupted -> large error ->
  the errors->LR/GMM detector separates them better. Distinguishability moves out of the
  activations and into the error, which is where the metric wants it.

Method (INLP / concept-erasure repurposed for unlearning):
  Each epoch, refit a linear probe on channel-mean activations at the erase blocks ->
  get unit direction w_i and the retain centroid projection mu_r,i. During the epoch,
  add an erase loss that drives each forget sample's standardised projection onto w_i
  toward mu_r,i. Refitting each epoch chases residual directions (iterative nullspace
  removal). Utility is held by a retain position loss + a light teacher activation anchor
  on retain samples. No position loss on forget (labels untrusted), unless --forget-target
  knn (exp05 kNN-corrected positions).

Stop signal: probe CV-AUC -> ~0.5 means the forget concept is no longer linearly readable.

Usage:
  python scripts/exp09_concept_erasure.py --forget-target none
  python scripts/exp09_concept_erasure.py --forget-target knn
Candidates are scored canonically with scripts/eval_robust.py --ckpt <path>.
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import load_model
from src.dataset import format_csi_for_cnn
from src.metrics import (get_predictions, prediction_errors, mia_accuracy,
                         gmm_threshold_predictions)

CFG = {
    "seed": 42,
    "batch_size": 64,
    "lr": 5e-5,
    "epochs": 20,
    "erase_idx": [14, 18, 21],     # post-ReLU blocks 4-6 (most probe-separable)
    "anchor_idx": [14, 18, 21],    # retain teacher anchor, same blocks
    "beta_anchor": 1.0,            # weight of retain activation anchor
    "lambda_erase": 1.0,           # weight of forget concept-erase loss
    "erase_warmup_epochs": 2,
    "delta_forget_pos": 1.0,       # weight of kNN-corrected forget pos loss (knn variant)
    "k_neighbours": 10,
    "retain_err_guard_m": 0.25,
    "eval_every": 2,
    "probe_cv_folds": 3,
}

DATA = ROOT / "data" / "public"
OUT_DIR = ROOT / "experiments" / "exp09_concept_erasure"
PSEUDO_PATH = ROOT / "experiments" / "exp06_direct_classifier" / "test_proba.npy"
EPS = 1e-8

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-in", default=str(ROOT / "data" / "baseline_cnn_task2.pth"))
parser.add_argument("--forget-target", choices=["none", "knn"], default="none")
parser.add_argument("--epochs", type=int, default=None)
args = parser.parse_args()
if args.epochs:
    CFG["epochs"] = args.epochs

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
print(f"Device: {DEVICE}   ckpt-in: {args.ckpt_in}   forget-target: {args.forget_target}")

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

class ActivationTap:
    """Forward hooks on model.features[i]; clones outputs (inplace-ReLU safe)."""

    def __init__(self, model, indices):
        self.acts = {}
        self.handles = [model.features[i].register_forward_hook(self._make(i))
                        for i in indices]

    def _make(self, i):
        def hook(_m, _inp, out):
            self.acts[i] = out.clone()
        return hook

    def clear(self):
        self.acts = {}

    def remove(self):
        for h in self.handles:
            h.remove()


def get_pos_corrected():
    """exp05 recipe: forget positions replaced by mean of 10 retain CSI-neighbours."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    print("Computing kNN-corrected forget positions (exp05 recipe)...")
    n = len(csi_train)
    mag = np.abs(csi_train.reshape(n, -1)).astype(np.float32)
    feats = PCA(n_components=64, random_state=CFG["seed"]).fit_transform(mag)
    nn_idx = (NearestNeighbors(n_neighbors=CFG["k_neighbours"])
              .fit(feats[retain_mask])
              .kneighbors(feats[forget_mask], return_distance=False))
    corrected = pos_train.copy()
    corrected[forget_mask] = pos_train[retain_mask][nn_idx].mean(axis=1)
    return corrected


def channel_mean_acts(model, idx):
    """Channel-mean (over spatial dims) activations at each block index, all of X_full."""
    tap = ActivationTap(model, idx)
    feats = {i: [] for i in idx}
    model.eval()
    try:
        with torch.no_grad():
            for s in range(0, len(X_full), 256):
                model(X_full[s:s + 256].to(DEVICE))
                for i in idx:
                    feats[i].append(tap.acts[i].mean(dim=(2, 3)).cpu().numpy())
                tap.clear()
    finally:
        tap.remove()
    return {i: np.concatenate(v) for i, v in feats.items()}


def fit_probe_directions(model):
    """Refit a linear probe per erase block; return frozen erase targets + CV-AUC.

    For each block i returns: per-channel mean/std (standardiser), unit direction w
    (probe coef in standardised space), retain centroid projection mu_r, and the
    projection std (for scale-free erase loss).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    feats = channel_mean_acts(model, CFG["erase_idx"])
    cv = StratifiedKFold(CFG["probe_cv_folds"], shuffle=True, random_state=CFG["seed"])
    targets, aucs = {}, {}
    for i in CFG["erase_idx"]:
        Fm = feats[i]
        scaler = StandardScaler().fit(Fm)
        Z = scaler.transform(Fm)
        lr = LogisticRegression(max_iter=1000).fit(Z, y)
        w = lr.coef_.ravel().astype(np.float32)
        w = w / (np.linalg.norm(w) + EPS)
        proj = Z @ w
        auc = float(cross_val_score(LogisticRegression(max_iter=1000), Z, y,
                                    cv=cv, scoring="roc_auc").mean())
        targets[i] = {
            "mean": torch.tensor(scaler.mean_, dtype=torch.float32, device=DEVICE),
            "std": torch.tensor(scaler.scale_, dtype=torch.float32, device=DEVICE),
            "w": torch.tensor(w, dtype=torch.float32, device=DEVICE),
            "mu_r": float(proj[retain_mask].mean()),
            "proj_std": float(proj.std() + EPS),
        }
        aucs[i] = auc
    return targets, aucs


def erase_loss(tap_acts, f_mask_batch, targets):
    """Drive forget samples' standardised projection onto w toward retain centroid mu_r."""
    L = None
    for i in CFG["erase_idx"]:
        t = targets[i]
        a = tap_acts[i][f_mask_batch].mean(dim=(2, 3))         # (Nf, C)
        z = (a - t["mean"]) / (t["std"] + EPS)
        proj = z @ t["w"]                                       # (Nf,)
        term = (((proj - t["mu_r"]) / t["proj_std"]) ** 2).mean()
        L = term if L is None else L + term
    return L / len(CFG["erase_idx"])


def quick_eval(model, with_test=False):
    err_tr = prediction_errors(get_predictions(model, X_full, device=DEVICE), pos_train)
    row = {
        "self_mia": mia_accuracy(err_tr, y, err_tr, y)["mia_accuracy"],
        "retain_err_m": float(err_tr[retain_mask].mean()),
        "forget_err_m": float(err_tr[forget_mask].mean()),
    }
    if with_test:
        err_te = prediction_errors(get_predictions(model, X_test, device=DEVICE), pos_test)
        gmm = gmm_threshold_predictions(err_te)
        row["gmm_test_forget_rate"] = gmm["forget_rate"]
        if PSEUDO is not None:
            row["pseudo_agreement"] = float((gmm["predictions"] == PSEUDO).mean())
    return row


def set_train_frozen_bn(model):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


# ---------------------------------------------------------------- run

def run():
    variant = args.forget_target
    student = load_model(args.ckpt_in, DEVICE)
    teacher = load_model(args.ckpt_in, DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    all_idx = sorted(set(CFG["erase_idx"]) | set(CFG["anchor_idx"]))
    tap_s, tap_t = ActivationTap(student, all_idx), ActivationTap(teacher, all_idx)

    Y_orig = torch.tensor(pos_train, dtype=torch.float32)
    Y_corr = (torch.tensor(get_pos_corrected(), dtype=torch.float32)
              if variant == "knn" else Y_orig)
    m_forget = torch.tensor(forget_mask)
    loader = DataLoader(TensorDataset(X_full, Y_orig, Y_corr, m_forget),
                        batch_size=CFG["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(CFG["seed"]))

    opt = torch.optim.Adam(student.parameters(), lr=CFG["lr"])
    best = {"self_mia": -1.0}
    best_path = OUT_DIR / f"model_best_{variant}.pth"
    log = []

    # baseline probe AUC before erasure
    _, auc0 = fit_probe_directions(student)
    print("  probe CV-AUC (baseline): "
          + "  ".join(f"blk{i}={auc0[i]:.3f}" for i in CFG["erase_idx"]))

    try:
        for epoch in range(1, CFG["epochs"] + 1):
            # refit probe directions on the *current* student (iterative nullspace removal)
            targets, aucs = fit_probe_directions(student)
            lam_t = CFG["lambda_erase"] * min(1.0, epoch / max(1, CFG["erase_warmup_epochs"]))

            set_train_frozen_bn(student)
            sums = {"pos_r": 0.0, "anc": 0.0, "erase": 0.0, "pos_f": 0.0}
            nb = 0
            for Xb, yo, yc, mb in loader:
                Xb, yo, yc, mb = Xb.to(DEVICE), yo.to(DEVICE), yc.to(DEVICE), mb.to(DEVICE)
                r, f = ~mb, mb
                tap_s.clear()
                tap_t.clear()
                pred = student(Xb)
                with torch.no_grad():
                    teacher(Xb)

                loss = pred.sum() * 0.0
                if r.any():
                    L_pos_r = F.mse_loss(pred[r], yo[r])
                    L_anc = pred.sum() * 0.0
                    for i in CFG["anchor_idx"]:
                        a = tap_s.acts[i][r].flatten(1)
                        a0 = tap_t.acts[i][r].flatten(1)
                        L_anc = L_anc + (((a - a0) ** 2).sum(1)
                                         / (a0.pow(2).sum(1) + EPS)).mean()
                    L_anc = L_anc / len(CFG["anchor_idx"])
                    loss = loss + L_pos_r + CFG["beta_anchor"] * L_anc
                    sums["pos_r"] += L_pos_r.item()
                    sums["anc"] += L_anc.item()
                if f.any():
                    L_erase = erase_loss(tap_s.acts, f, targets)
                    loss = loss + lam_t * L_erase
                    sums["erase"] += L_erase.item()
                    if variant == "knn":
                        L_pos_f = F.mse_loss(pred[f], yc[f])
                        loss = loss + CFG["delta_forget_pos"] * L_pos_f
                        sums["pos_f"] += L_pos_f.item()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                nb += 1

            entry = {"epoch": epoch, "lambda_t": lam_t,
                     "probe_auc": {str(i): aucs[i] for i in CFG["erase_idx"]},
                     **{k: v / nb for k, v in sums.items()}}
            print(f"  ep {epoch:2d}  pos_r={entry['pos_r']:.4f}  anc={entry['anc']:.4f}  "
                  f"erase={entry['erase']:.4f}  pos_f={entry['pos_f']:.4f}  "
                  + "probeAUC[" + ",".join(f"{aucs[i]:.3f}" for i in CFG["erase_idx"]) + "]")

            if epoch % CFG["eval_every"] == 0 or epoch == CFG["epochs"]:
                student.eval()
                ev = quick_eval(student, with_test=True)
                entry.update(ev)
                print(f"      selfMIA={ev['self_mia']:.4f}  retain={ev['retain_err_m']:.4f}m  "
                      f"forget={ev['forget_err_m']:.4f}m  gmm_rate={ev['gmm_test_forget_rate']:.3f}"
                      + (f"  pseudo={ev['pseudo_agreement']:.4f}" if "pseudo_agreement" in ev else ""))
                if (ev["retain_err_m"] <= CFG["retain_err_guard_m"]
                        and ev["self_mia"] > best["self_mia"]):
                    best = {**ev, "epoch": epoch}
                    torch.save(student.state_dict(), best_path)
                    print(f"      saved -> {best_path}")
            log.append(entry)
    finally:
        tap_s.remove()
        tap_t.remove()

    out = {"ckpt_in": args.ckpt_in, "variant": variant, "config": CFG,
           "baseline_probe_auc": {str(i): auc0[i] for i in CFG["erase_idx"]},
           "best": best, "log": log}
    (OUT_DIR / f"train_log_{variant}.json").write_text(json.dumps(out, indent=2))
    path = OUT_DIR / "metrics.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged[variant] = {k: v for k, v in out.items() if k != "log"}
    path.write_text(json.dumps(merged, indent=2))

    if best["self_mia"] < 0:
        print("  !! no epoch passed the retain-error guard — nothing saved")
    else:
        print(f"\n  best epoch {best['epoch']}: selfMIA={best['self_mia']:.4f}  "
              f"retain={best['retain_err_m']:.4f}m  forget={best['forget_err_m']:.4f}m  "
              f"gmm_rate={best['gmm_test_forget_rate']:.3f}")
        print(f"  Run: python scripts/eval_robust.py --ckpt {best_path}")


if __name__ == "__main__":
    run()
