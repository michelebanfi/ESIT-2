"""Experiment 08 — activation-trajectory unlearning.

Idea: information flowing through the CNN is a trajectory; detach the flow that the
corrupted (forget) samples rely on. Three parts:

  --probe    diagnostic: where does forget-specific signal live? Per-layer activation
             separability (Cohen's d + linear probe AUC) and Fisher parameter importance
             computed separately on the forget set (with its original corrupted labels —
             how its influence entered the weights) and on the retain set.
  --ssd      Selective Synaptic Dampening (training-free): where a parameter's forget
             importance dominates its retain importance, shrink it. Grid over
             (alpha, lambda); retain-only BatchNorm recalibration after dampening.
  --diverge  trajectory-divergence finetune: anchor retain activations to a frozen
             teacher, push forget activations off their baseline trajectories.
             Divergence acts on pre-ReLU BatchNorm outputs (ReLU outputs are
             non-negative, so "far away" has a trivial all-zero solution); a hinged
             cosine saturates once orthogonal and a norm-keeper forbids collapse.

Usage:
  python scripts/exp08_activation_unlearning.py --probe
  python scripts/exp08_activation_unlearning.py --ssd [--ckpt-in <path>]
  python scripts/exp08_activation_unlearning.py --diverge --forget-target {none,knn}

Candidates are scored canonically with scripts/eval_robust.py --ckpt <path>.
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import load_model
from src.dataset import format_csi_for_cnn
from src.metrics import (get_predictions, prediction_errors, mia_accuracy,
                         gmm_threshold_predictions, localization_stats)

CFG = {
    "seed": 42,
    "batch_size": 64,
    # --- probe ---
    "probe_relu_idx": [2, 6, 10, 14, 18, 21],
    "fisher_batch_size": 64,
    # --- ssd ---
    "ssd_alphas": [1.0, 2.0, 5.0, 10.0],
    "ssd_lambdas": [0.1, 0.5, 1.0],
    "ssd_exclude_bn": True,
    "ssd_exclude_head": False,
    "retain_err_guard_m": 0.25,
    "gmm_rate_range": [0.40, 0.60],
    # --- diverge ---
    "div_bn_idx": [13, 17, 20],      # pre-ReLU BN outputs, blocks 4-6
    "anchor_relu_idx": [14, 18, 21],  # post-ReLU outputs, blocks 4-6
    "lr": 5e-5,
    "epochs": 12,
    "beta_anchor": 1.0,
    "gamma_diverge": 0.5,
    "gamma_warmup_epochs": 2,
    "cos_margin": 0.0,
    "norm_keep_weight": 0.1,
    "delta_forget_pos": 1.0,
    "k_neighbours": 10,
    "eval_every": 3,
}

DATA = ROOT / "data" / "public"
OUT_DIR = ROOT / "experiments" / "exp08_activation_unlearning"
PSEUDO_PATH = ROOT / "experiments" / "exp06_direct_classifier" / "test_proba.npy"
EPS = 1e-8

parser = argparse.ArgumentParser()
parser.add_argument("--probe", action="store_true")
parser.add_argument("--ssd", action="store_true")
parser.add_argument("--diverge", action="store_true")
parser.add_argument("--ckpt-in", default=str(ROOT / "data" / "baseline_cnn_task2.pth"))
parser.add_argument("--forget-target", choices=["none", "knn"], default="none")
parser.add_argument("--epochs", type=int, default=None, help="override CFG['epochs']")
args = parser.parse_args()
if args.epochs:
    CFG["epochs"] = args.epochs
if not (args.probe or args.ssd or args.diverge):
    parser.error("pick at least one of --probe / --ssd / --diverge")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(CFG["seed"])
CKPT_STEM = Path(args.ckpt_in).stem
print(f"Device: {DEVICE}   ckpt-in: {args.ckpt_in}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
for sub in ("probe", "ssd", "diverge"):
    (OUT_DIR / sub).mkdir(exist_ok=True)

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


# ---------------------------------------------------------------- shared helpers

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


_pos_corrected_cache = None


def get_pos_corrected():
    """exp05 recipe: forget positions replaced by mean of 10 retain CSI-neighbours."""
    global _pos_corrected_cache
    if _pos_corrected_cache is None:
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
        _pos_corrected_cache = corrected
    return _pos_corrected_cache


def quick_eval(model, with_test=True):
    """Self-MIA + retain/forget errors on train; optional GMM/pseudo on test."""
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


def fisher_importance(model, X, Y, tag):
    """Batch-Fisher: squared grads of MSE accumulated over batches (MPS-safe)."""
    imp = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    crit = nn.MSELoss()
    model.eval()  # no BN running-stat updates
    bs = CFG["fisher_batch_size"]
    nb = 0
    for i in range(0, len(X), bs):
        model.zero_grad(set_to_none=True)
        loss = crit(model(X[i:i + bs].to(DEVICE)), Y[i:i + bs].to(DEVICE))
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                imp[n] += p.grad.detach() ** 2
        nb += 1
    model.zero_grad(set_to_none=True)
    print(f"  Fisher[{tag}]: {nb} batches")
    return {n: (v / nb).cpu() for n, v in imp.items()}


def get_fisher(model):
    """Load cached Fisher dicts for this checkpoint or compute them."""
    f_path = OUT_DIR / "probe" / f"fisher_forget_{CKPT_STEM}.pth"
    r_path = OUT_DIR / "probe" / f"fisher_retain_{CKPT_STEM}.pth"
    if f_path.exists() and r_path.exists():
        print("Loading cached Fisher importances...")
        return torch.load(f_path, weights_only=True), torch.load(r_path, weights_only=True)
    Y_t = torch.tensor(pos_train, dtype=torch.float32)
    imp_f = fisher_importance(model, X_full[forget_mask], Y_t[forget_mask], "forget")
    imp_r = fisher_importance(model, X_full[retain_mask], Y_t[retain_mask], "retain")
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


def set_train_frozen_bn(model):
    """Train mode but BN running stats frozen (affine params still learn)."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def update_metrics(key, payload):
    path = OUT_DIR / "metrics.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged[key] = payload
    path.write_text(json.dumps(merged, indent=2))
    print(f"metrics.json <- {key}")


# ---------------------------------------------------------------- --probe

def run_probe():
    print("\n=== PROBE: activation separability + Fisher importance ===")
    model = load_model(args.ckpt_in, DEVICE)
    model.eval()

    # per-layer channel-mean activations
    idx = CFG["probe_relu_idx"]
    tap = ActivationTap(model, idx)
    feats = {i: [] for i in idx}
    try:
        with torch.no_grad():
            for s in range(0, len(X_full), 256):
                model(X_full[s:s + 256].to(DEVICE))
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
        pooled = np.sqrt(((n_f - 1) * s_f**2 + (n_r - 1) * s_r**2) / (n_f + n_r - 2))
        d = (mu_f - mu_r) / (pooled + EPS)
        np.save(OUT_DIR / "probe" / f"cohens_d_layer{i}.npy", d)
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
    (OUT_DIR / "probe" / "layer_separability.json").write_text(json.dumps(sep, indent=2))

    # Fisher importance, forget (original corrupted labels) vs retain
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
    (OUT_DIR / "probe" / "fisher_layer_summary.json").write_text(json.dumps(summary, indent=2))
    print("  Fisher ratio I_f/I_r > 5, per parameter tensor:")
    for n, s in summary.items():
        if s["frac_ratio_gt_5"] > 0.01:
            print(f"    {n:35s} frac>5 = {s['frac_ratio_gt_5']:.3f}")
    update_metrics("probe", {"ckpt_in": args.ckpt_in, "layer_separability": sep})


# ---------------------------------------------------------------- --ssd

def run_ssd():
    print("\n=== SSD: selective synaptic dampening ===")
    model = load_model(args.ckpt_in, DEVICE)
    imp_f, imp_r = get_fisher(model)
    base_sd = copy.deepcopy(model.state_dict())
    skip = bn_param_names(model) if CFG["ssd_exclude_bn"] else set()
    if CFG["ssd_exclude_head"]:
        skip |= {n for n in base_sd if n.startswith("regression_head")}

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
                  f"forget={row['forget_err_m']:.4f}m  gmm_rate={row['gmm_test_forget_rate']:.3f}"
                  + (f"  pseudo={row['pseudo_agreement']:.4f}" if "pseudo_agreement" in row else ""))

    lo, hi = CFG["gmm_rate_range"]
    ok = [r for r in rows
          if r["retain_err_m"] <= CFG["retain_err_guard_m"] and lo <= r["gmm_test_forget_rate"] <= hi]
    key = "pseudo_agreement" if (ok and "pseudo_agreement" in ok[0]) else "self_mia"
    best = max(ok, key=lambda r: r[key]) if ok else None

    out = {"ckpt_in": args.ckpt_in, "grid": rows, "selection_key": key, "best": best}
    (OUT_DIR / "ssd" / f"grid_results_{CKPT_STEM}.json").write_text(json.dumps(out, indent=2))
    if best is None:
        print("  !! no grid point passed the guards (retain err / GMM rate) — nothing saved")
    else:
        # re-apply the winning setting and save weights
        sd = copy.deepcopy(base_sd)
        for n in sd:
            if n in skip or n not in imp_f:
                continue
            If, Ir = imp_f[n].to(DEVICE), imp_r[n].to(DEVICE)
            sel = If > best["alpha"] * Ir
            if sel.any():
                sd[n][sel] *= torch.clamp(best["lambda"] * Ir / (If + 1e-12), max=1.0)[sel]
        model.load_state_dict(sd)
        recalibrate_bn(model)
        best_path = OUT_DIR / "ssd" / f"model_ssd_best_{CKPT_STEM}.pth"
        torch.save(model.state_dict(), best_path)
        print(f"  best: alpha={best['alpha']} lambda={best['lambda']} "
              f"({key}={best[key]:.4f})  saved -> {best_path}")
    update_metrics(f"ssd_{CKPT_STEM}", out)


# ---------------------------------------------------------------- --diverge

def run_diverge():
    variant = args.forget_target
    print(f"\n=== DIVERGE: trajectory-divergence finetune (forget-target={variant}) ===")
    student = load_model(args.ckpt_in, DEVICE)
    teacher = load_model(args.ckpt_in, DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    div_idx, anc_idx = CFG["div_bn_idx"], CFG["anchor_relu_idx"]
    all_idx = sorted(set(div_idx) | set(anc_idx))
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
    best_path = OUT_DIR / "diverge" / f"model_best_{variant}.pth"
    log = []

    try:
        for epoch in range(1, CFG["epochs"] + 1):
            set_train_frozen_bn(student)
            gamma_t = CFG["gamma_diverge"] * min(1.0, epoch / max(1, CFG["gamma_warmup_epochs"]))
            sums = {"pos_r": 0.0, "anc": 0.0, "div": 0.0, "norm": 0.0, "pos_f": 0.0}
            ratio_sum, ratio_n, nb = 0.0, 0, 0

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
                    for i in anc_idx:
                        a = tap_s.acts[i][r].flatten(1)
                        a0 = tap_t.acts[i][r].flatten(1)
                        L_anc = L_anc + (((a - a0) ** 2).sum(1)
                                         / (a0.pow(2).sum(1) + EPS)).mean()
                    L_anc = L_anc / len(anc_idx)
                    loss = loss + L_pos_r + CFG["beta_anchor"] * L_anc
                    sums["pos_r"] += L_pos_r.item()
                    sums["anc"] += L_anc.item()
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
                    L_div, L_norm = L_div / len(div_idx), L_norm / len(div_idx)
                    loss = loss + gamma_t * L_div + CFG["norm_keep_weight"] * L_norm
                    sums["div"] += L_div.item()
                    sums["norm"] += L_norm.item()
                    if variant == "knn":
                        L_pos_f = F.mse_loss(pred[f], yc[f])
                        loss = loss + CFG["delta_forget_pos"] * L_pos_f
                        sums["pos_f"] += L_pos_f.item()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                nb += 1

            entry = {"epoch": epoch, "gamma_t": gamma_t,
                     **{k: v / nb for k, v in sums.items()},
                     "forget_norm_ratio": ratio_sum / max(1, ratio_n)}
            collapse = " !! norm ratio < 0.5 — collapse warning" \
                if entry["forget_norm_ratio"] < 0.5 else ""
            print(f"  ep {epoch:2d}  pos_r={entry['pos_r']:.4f}  anc={entry['anc']:.4f}  "
                  f"div={entry['div']:.4f}  norm={entry['norm']:.4f}  "
                  f"pos_f={entry['pos_f']:.4f}  |a|/|a0|={entry['forget_norm_ratio']:.3f}"
                  f"{collapse}")

            if epoch % CFG["eval_every"] == 0 or epoch == CFG["epochs"]:
                student.eval()
                ev = quick_eval(student, with_test=False)
                entry.update(ev)
                print(f"      selfMIA={ev['self_mia']:.4f}  retain={ev['retain_err_m']:.4f}m  "
                      f"forget={ev['forget_err_m']:.4f}m")
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
           "best": best, "log": log}
    (OUT_DIR / "diverge" / f"train_log_{variant}.json").write_text(json.dumps(out, indent=2))
    if best["self_mia"] < 0:
        print("  !! no epoch passed the retain-error guard — nothing saved")
    else:
        print(f"  best epoch {best['epoch']}: selfMIA={best['self_mia']:.4f}  "
              f"retain={best['retain_err_m']:.4f}m  forget={best['forget_err_m']:.4f}m")
        print(f"  Run: python scripts/eval_robust.py --ckpt {best_path}")
    update_metrics(f"diverge_{variant}", {k: v for k, v in out.items() if k != "log"})


if __name__ == "__main__":
    if args.probe:
        run_probe()
    if args.ssd:
        run_ssd()
    if args.diverge:
        run_diverge()
