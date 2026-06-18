"""Generate all presentation figures for the activation-trajectory unlearning paper.

Produces 11 vector PDFs in figures/.  Reuses cached JSON/NPY artifacts where available;
only the activation scatter / before-after / cosine plots need fresh forward passes
over the train set (baseline + unlearned model; ~2 min on M1 with MPS).

Output (figures/):
  01_contamination_shift.pdf    kNN-consistency shift: forget vs retain
  02_cnn_architecture.pdf       DichasusPositionPredictor block diagram
  03_probe_auc_by_block.pdf     5-fold linear probe AUC per block
  04_activation_scatter.pdf     PCA-2D scatter of block-5 acts (baseline)
  05_cohens_d_by_block.pdf      |Cohen's d| distribution per channel per block
  06_fisher_ratio_by_block.pdf  fraction of weights with I_f/I_r > 5 per block
  07_ssd_negative_result.pdf    SSD grid (retain_err vs GMM-forget-rate)
  08_effect_before_after.pdf    PCA-2D baseline vs unlearned (forget displaced)
  09_effect_cosine_hist.pdf     cos(student, teacher) BN acts: forget ≈ 0, retain ≈ 1
  10_training_dynamics.pdf      per-epoch loss terms + eval metrics (beta=3 run)
  11_beta_anchor_ablation.pdf   retain_err & LR forget-rate vs beta
  12_error_distribution.pdf     per-sample error histogram + LR boundary

Usage:
  python scripts/probing/make_figures.py [--ckpt-base path] [--ckpt-unlearned path]
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from src.model import load_model
from src.dataset import format_csi_for_cnn
from src.metrics import get_predictions, prediction_errors, mia_accuracy
from src.activations import ActivationTap

# ── argument defaults ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-base",
                    default=str(ROOT / "data" / "baseline_cnn_task2.pth"))
parser.add_argument("--ckpt-unlearned",
                    default=str(ROOT / "experiments" / "diverge" / "model_official_knn_b3.pth"))
args = parser.parse_args()

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PROBE_DIR = ROOT / "experiments" / "probe"
DIVERGE_DIR = ROOT / "experiments" / "diverge"
CONT_DIR = ROOT / "experiments" / "probe_contamination"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "public"

# ── matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

C_FORGET = "#e05252"
C_RETAIN = "#4a90d9"
C_TEACHER = "#aaaaaa"

BLOCK_LABELS = ["B1\n(32ch)", "B2\n(64ch)", "B3\n(128ch)",
                 "B4\n(256ch)", "B5\n(512ch)", "B6\n(512ch)"]
RELU_IDX = [2, 6, 10, 14, 18, 21]   # post-ReLU probe hook indices
BN_IDX   = [13, 17, 20]             # pre-ReLU BN diverge hook indices

print(f"Device: {DEVICE}")

# ── load data (once) ─────────────────────────────────────────────────────────
print("Loading train data...")
csi_train = np.load(DATA / "task2_train_csi.npy")
pos_train  = np.load(DATA / "task2_train_positions.npy")[:, :2]
meta_train = pd.read_csv(DATA / "task2_train_metadata.csv")
y          = meta_train["is_forget"].values
forget_mask = y == 1
retain_mask = ~forget_mask
N = len(y)
print(f"  N={N}  retain={retain_mask.sum()}  forget={forget_mask.sum()}")

X_full = format_csi_for_cnn(csi_train)


# ══════════════════════════════════════════════════════════════════════════════
# Figure helpers
# ══════════════════════════════════════════════════════════════════════════════

def save(name: str, fig: plt.Figure):
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def get_channel_mean_acts(model, indices):
    """Return {idx: (N, C) channel-mean activations} for all indices."""
    tap = ActivationTap(model, indices)
    feats = {i: [] for i in indices}
    model.eval()
    try:
        with torch.no_grad():
            for s in range(0, N, 256):
                model(X_full[s:s+256].to(DEVICE))
                for i in indices:
                    feats[i].append(tap.acts[i].mean(dim=(2, 3)).cpu().numpy())
                tap.clear()
    finally:
        tap.remove()
    return {i: np.concatenate(v) for i, v in feats.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 01  Contamination shift
# ══════════════════════════════════════════════════════════════════════════════
print("\n[01] contamination_shift")
try:
    res = json.loads((CONT_DIR / "results.json").read_text())
    # compute consistency from scratch so we have the full distribution
    raise FileNotFoundError("need full distribution")
except Exception:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    mag = np.abs(csi_train.reshape(N, -1)).astype(np.float32)
    pca64 = PCA(n_components=64, random_state=42).fit_transform(mag)
    nn_all = NearestNeighbors(n_neighbors=11).fit(pca64)
    _, idx = nn_all.kneighbors(pca64)
    idx = idx[:, 1:]
    consistency = np.linalg.norm(pos_train[idx] - pos_train[:, None, :], axis=2).mean(axis=1)

fig, ax = plt.subplots(figsize=(6, 3.5))
bins = np.linspace(0, 4, 60)
ax.hist(consistency[retain_mask], bins=bins, color=C_RETAIN, alpha=0.7, label="Retain", density=True)
ax.hist(consistency[forget_mask], bins=bins, color=C_FORGET, alpha=0.7, label="Forget", density=True)
ax.axvline(consistency[retain_mask].mean(), color=C_RETAIN, lw=2, ls="--",
           label=f"Retain mean {consistency[retain_mask].mean():.2f} m")
ax.axvline(consistency[forget_mask].mean(), color=C_FORGET, lw=2, ls="--",
           label=f"Forget mean {consistency[forget_mask].mean():.2f} m")
ax.set_xlabel("Mean kNN position-consistency (m)")
ax.set_ylabel("Density")
ax.set_title("Contamination diagnosis: forget labels are corrupted")
ax.legend(fontsize=9)
fig.tight_layout()
save("01_contamination_shift.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
# 03  Probe AUC by block
# ══════════════════════════════════════════════════════════════════════════════
print("[03] probe_auc_by_block")
sep = json.loads((PROBE_DIR / "layer_separability.json").read_text())
aucs = [sep[str(i)]["probe_auc_cv"] for i in RELU_IDX]

fig, ax = plt.subplots(figsize=(6, 3.5))
bars = ax.bar(range(6), aucs, color=C_RETAIN, alpha=0.85, zorder=3)
ax.set_xticks(range(6))
ax.set_xticklabels(BLOCK_LABELS)
ax.set_ylim(0.93, 1.00)
ax.set_ylabel("5-fold CV ROC-AUC")
ax.set_title("Linear probe AUC on channel-mean activations by block")
ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.4)
ax.grid(axis="y", alpha=0.4, zorder=0)
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, auc + 0.001, f"{auc:.3f}",
            ha="center", va="bottom", fontsize=9)
fig.tight_layout()
save("03_probe_auc_by_block.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
# 04  Activation scatter (block 5, baseline) — LDA discriminant projection
# ══════════════════════════════════════════════════════════════════════════════
print("[04] activation_scatter (loading baseline model...)")
model_base = load_model(args.ckpt_base, DEVICE)
feats_base = get_channel_mean_acts(model_base, [18])   # block-5 post-ReLU

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA as skPCA

lda = LinearDiscriminantAnalysis()
proj_lda_base = lda.fit_transform(feats_base[18], y)[:, 0]   # (N,) — single discriminant

fig, ax = plt.subplots(figsize=(6, 4))
bins = np.linspace(proj_lda_base.min() - 0.2, proj_lda_base.max() + 0.2, 70)
ax.hist(proj_lda_base[retain_mask], bins=bins, color=C_RETAIN, alpha=0.7, density=True,
        label=f"Retain (n={retain_mask.sum()})")
ax.hist(proj_lda_base[forget_mask], bins=bins, color=C_FORGET, alpha=0.7, density=True,
        label=f"Forget (n={forget_mask.sum()})")
ax.axvline(proj_lda_base[retain_mask].mean(), color=C_RETAIN, lw=2, ls="--",
           label=f"Retain mean {proj_lda_base[retain_mask].mean():.2f}")
ax.axvline(proj_lda_base[forget_mask].mean(), color=C_FORGET, lw=2, ls="--",
           label=f"Forget mean {proj_lda_base[forget_mask].mean():.2f}")
ax.set_xlabel("LDA discriminant projection (block-5 channel-mean activations)")
ax.set_ylabel("Density")
ax.set_title("Block-5 activations (baseline): forget / retain are linearly separable\n"
             f"LDA discriminant — 5-fold probe AUC = {aucs[4]:.3f}")
ax.legend(fontsize=9, framealpha=0.9)
fig.tight_layout()
save("04_activation_scatter.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
# 05  Cohen's d distribution per block
# ══════════════════════════════════════════════════════════════════════════════
print("[05] cohens_d_by_block")
all_d = {i: np.load(PROBE_DIR / f"cohens_d_layer{i}.npy") for i in RELU_IDX}

fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True)
for ax, i, label in zip(axes.flat, RELU_IDX, BLOCK_LABELS):
    d = np.abs(all_d[i])
    ax.hist(d, bins=30, color=C_RETAIN, alpha=0.8, density=True)
    ax.axvline(d.mean(), color="k", lw=1.5, ls="--", label=f"mean {d.mean():.2f}")
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.set_xlabel("|Cohen's d|")
    ax.set_ylabel("Density")
fig.suptitle("Channel-level separability (|Cohen's d|, forget vs retain) by block", y=1.01)
fig.tight_layout()
save("05_cohens_d_by_block.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
# 06  Fisher ratio per block
# ══════════════════════════════════════════════════════════════════════════════
print("[06] fisher_ratio_by_block")
fisher_sum = json.loads((PROBE_DIR / "fisher_layer_summary.json").read_text())

# Aggregate by block: sum (numel * frac>5) / total_numel per block
block_conv_keys = [
    ["features.0.weight", "features.0.bias"],    # block 1
    ["features.4.weight", "features.4.bias"],    # block 2
    ["features.8.weight", "features.8.bias"],    # block 3
    ["features.12.weight", "features.12.bias"],  # block 4
    ["features.16.weight", "features.16.bias"],  # block 5
    ["features.19.weight", "features.19.bias"],  # block 6
]
frac5_per_block = []
for keys in block_conv_keys:
    total = sum(fisher_sum[k]["numel"] for k in keys if k in fisher_sum)
    weighted = sum(fisher_sum[k]["numel"] * fisher_sum[k]["frac_ratio_gt_5"]
                   for k in keys if k in fisher_sum)
    frac5_per_block.append(weighted / total if total > 0 else 0.0)

fig, ax = plt.subplots(figsize=(6, 3.5))
bars = ax.bar(range(6), frac5_per_block, color="#f0a830", alpha=0.9, zorder=3)
ax.set_xticks(range(6))
ax.set_xticklabels(BLOCK_LABELS)
ax.set_ylabel("Fraction of weights with $I_f/I_r > 5$")
ax.set_title("Forget-dominant Fisher mass per block\n"
             "(high = forget influence concentrated here; SSD target)")
ax.grid(axis="y", alpha=0.4, zorder=0)
for bar, v in zip(bars, frac5_per_block):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.2f}",
            ha="center", va="bottom", fontsize=9)
fig.tight_layout()
save("06_fisher_ratio_by_block.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
# 07  SSD negative result
# ══════════════════════════════════════════════════════════════════════════════
print("[07] ssd_negative_result")
ssd = json.loads((PROBE_DIR / "ssd_grid_results_baseline.json").read_text())
grid = ssd["grid"]
alphas_all = sorted(set(r["alpha"] for r in grid))
colors_map = {1.0: "#1a1a2e", 2.0: "#16213e", 5.0: "#0f3460", 10.0: "#533483"}
markers_map = {1.0: "o", 2.0: "s", 5.0: "^", 10.0: "D"}

fig, ax = plt.subplots(figsize=(6, 4))
for alpha in alphas_all:
    rows = [r for r in grid if r["alpha"] == alpha]
    retain_errs = [r["retain_err_m"] for r in rows]
    gmm_rates = [r["gmm_test_forget_rate"] for r in rows]
    ax.scatter(retain_errs, gmm_rates, s=70, label=f"α={alpha}",
               color=colors_map[alpha], marker=markers_map[alpha], zorder=3)

# feasibility box
ax.axhspan(0.40, 0.60, color="green", alpha=0.08, label="Target region (GMM)")
ax.axvline(0.25, color="red", ls="--", lw=1.5, alpha=0.7, label="Retain guard 0.25 m")
ax.set_xlabel("Retain mean error (m)")
ax.set_ylabel("GMM test forget rate")
ax.set_title("SSD grid: all configurations destroy retain utility\n"
             "(forget/retain Fisher importances are entangled)")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3, zorder=0)
fig.tight_layout()
save("07_ssd_negative_result.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
# 08  Before/after: LDA projection shift + activation displacement norm
# ══════════════════════════════════════════════════════════════════════════════
print("[08] effect_before_after (loading unlearned model...)")
model_ul = load_model(args.ckpt_unlearned, DEVICE)
feats_ul = get_channel_mean_acts(model_ul, [18])   # block-5 post-ReLU

# Project unlearned activations into the baseline LDA discriminant space
proj_lda_ul = lda.transform(feats_ul[18])[:, 0]   # (N,)

# Per-sample L2 displacement in activation space (baseline → unlearned)
delta = feats_ul[18] - feats_base[18]              # (N, C)
disp  = np.linalg.norm(delta, axis=1)              # (N,)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# ── Left: LDA projection distributions before and after ──────────────────────
ax = axes[0]
lo = min(proj_lda_base.min(), proj_lda_ul.min()) - 0.1
hi = max(proj_lda_base.max(), proj_lda_ul.max()) + 0.1
bins_lda = np.linspace(lo, hi, 65)

ax.hist(proj_lda_base[forget_mask], bins=bins_lda, color=C_FORGET, alpha=0.4, density=True,
        histtype="stepfilled", label="Forget — baseline")
ax.hist(proj_lda_ul[forget_mask],   bins=bins_lda, color=C_FORGET, alpha=0.75, density=True,
        histtype="step", lw=2.5, label="Forget — unlearned")
ax.hist(proj_lda_base[retain_mask], bins=bins_lda, color=C_RETAIN, alpha=0.4, density=True,
        histtype="stepfilled", label="Retain — baseline")
ax.hist(proj_lda_ul[retain_mask],   bins=bins_lda, color=C_RETAIN, alpha=0.75, density=True,
        histtype="step", lw=2.5, label="Retain — unlearned")
ax.set_xlabel("LDA discriminant projection (block-5 channel-mean acts)")
ax.set_ylabel("Density")
ax.set_title("Forget distribution shifts along\nthe discriminant axis; retain stays put")
ax.legend(fontsize=8, framealpha=0.9)

# ── Right: activation displacement norm distribution ─────────────────────────
ax = axes[1]
p99 = np.percentile(disp, 99)
bins_d = np.linspace(0, p99 * 1.08, 65)
ax.hist(disp[retain_mask], bins=bins_d, color=C_RETAIN, alpha=0.7, density=True,
        label=f"Retain  (mean {disp[retain_mask].mean():.1f})")
ax.hist(disp[forget_mask], bins=bins_d, color=C_FORGET, alpha=0.7, density=True,
        label=f"Forget  (mean {disp[forget_mask].mean():.1f})")
ax.set_xlabel(r"$\|\mathbf{a}_{unlearned} - \mathbf{a}_{baseline}\|_2$  (block-5 acts)")
ax.set_ylabel("Density")
ax.set_title("Forget activations are displaced further\nfrom the baseline trajectory")
ax.legend(fontsize=9, framealpha=0.9)

fig.suptitle("Effect of activation-trajectory divergence on block-5 representations",
             fontsize=12, y=1.01)
fig.tight_layout()
save("08_effect_before_after.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
# 09  Cosine similarity histogram (BN layers, baseline vs unlearned)
# ══════════════════════════════════════════════════════════════════════════════
print("[09] effect_cosine_hist (BN-layer forward passes...)")
feats_base_bn = get_channel_mean_acts(model_base, BN_IDX)
feats_ul_bn   = get_channel_mean_acts(model_ul,   BN_IDX)

def cosine_sim_flat(A, B):
    """Per-sample cosine similarity between two (N, C) activation matrices."""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (An * Bn).sum(axis=1)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for ax, i, label in zip(axes, BN_IDX,
                          ["Block 4 BN", "Block 5 BN", "Block 6 BN"]):
    cos_f = cosine_sim_flat(feats_ul_bn[i][forget_mask], feats_base_bn[i][forget_mask])
    cos_r = cosine_sim_flat(feats_ul_bn[i][retain_mask], feats_base_bn[i][retain_mask])
    bins = np.linspace(-1, 1, 50)
    ax.hist(cos_f, bins=bins, color=C_FORGET, alpha=0.7, density=True, label="Forget")
    ax.hist(cos_r, bins=bins, color=C_RETAIN, alpha=0.7, density=True, label="Retain")
    ax.axvline(cos_f.mean(), color=C_FORGET, lw=1.8, ls="--",
               label=f"F mean {cos_f.mean():.2f}")
    ax.axvline(cos_r.mean(), color=C_RETAIN, lw=1.8, ls="--",
               label=f"R mean {cos_r.mean():.2f}")
    ax.set_title(label)
    ax.set_xlabel(r"$\cos(\mathbf{a}_s, \mathbf{a}_t)$")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Density")
fig.suptitle("Cosine similarity (student vs teacher) after diverge unlearning\n"
             "Forget: pushed toward 0 / negative.  Retain: anchored near 1.", y=1.02)
fig.tight_layout()
save("09_effect_cosine_hist.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
# 10  Training dynamics (beta=3 run)
# ══════════════════════════════════════════════════════════════════════════════
print("[10] training_dynamics")
log_data = json.loads((DIVERGE_DIR / "train_log_knn_b3.json").read_text())
log = log_data["log"]
best_epoch = log_data["best"].get("epoch", 24)

epochs = [e["epoch"] for e in log]
pos_r  = [e["pos_r"] for e in log]
anc    = [e["anc"] for e in log]
div    = [e["div"] for e in log]
norm   = [e["norm"] for e in log]
pos_f  = [e["pos_f"] for e in log]
nr     = [e["forget_norm_ratio"] for e in log]

eval_epochs = [e["epoch"] for e in log if "self_mia" in e]
ret_errs = [e["retain_err_m"] for e in log if "retain_err_m" in e]
fgt_errs = [e["forget_err_m"] for e in log if "forget_err_m" in e]
lr_accs  = [e.get("lr_pseudo_agreement", float("nan")) for e in log if "self_mia" in e]

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs, pos_r,  label="$L_{pos,R}$", lw=1.8)
ax1.plot(epochs, anc,    label="$L_{anc}$",   lw=1.8)
ax1.plot(epochs, div,    label="$L_{div}$",   lw=1.8)
ax1.plot(epochs, norm,   label="$L_{norm}$",  lw=1.8)
ax1.plot(epochs, pos_f,  label="$L_{pos,F}$", lw=1.8)
ax1.axvline(best_epoch, color="k", ls="--", lw=1.0, alpha=0.7, label=f"ep {best_epoch} (selected)")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss component")
ax1.set_title("Loss terms"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, nr, color="purple", lw=1.8)
ax2.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.5, label="no change")
ax2.axhline(0.5, color="red", ls=":", lw=1.0, alpha=0.7, label="collapse warning")
ax2.axvline(best_epoch, color="k", ls="--", lw=1.0, alpha=0.7)
ax2.set_xlabel("Epoch"); ax2.set_ylabel(r"$\|\mathbf{a}_s\|/\|\mathbf{a}_t\|$")
ax2.set_title("Forget norm ratio (no collapse expected)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(eval_epochs, ret_errs, "o-", color=C_RETAIN, lw=2, ms=5, label="Retain error")
ax3.plot(eval_epochs, fgt_errs, "o-", color=C_FORGET, lw=2, ms=5, label="Forget error")
ax3.axvline(best_epoch, color="k", ls="--", lw=1.0, alpha=0.7)
ax3.set_xlabel("Epoch"); ax3.set_ylabel("Mean error (m)")
ax3.set_title("Train errors vs. original labels"); ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(eval_epochs, lr_accs, "o-", color="navy", lw=2, ms=5)
ax4.axvline(best_epoch, color="k", ls="--", lw=1.0, alpha=0.7, label=f"ep {best_epoch}")
ax4.set_xlabel("Epoch"); ax4.set_ylabel("LR~acc (offline Kaggle proxy)")
ax4.set_title("Official-LR test accuracy (proxy)"); ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

fig.suptitle(r"Training dynamics — diverge, $\beta=3$, forget-target=knn", fontsize=13)
save("10_training_dynamics.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
# 11  Beta anchor ablation
# ══════════════════════════════════════════════════════════════════════════════
print("[11] beta_anchor_ablation")
betas, retain_errs_b, fgt_rates_b = [], [], []
for b_tag in ["0.5", "1", "1.5", "2", "3", "4"]:
    p = DIVERGE_DIR / f"train_log_knn_b{b_tag}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    best = d.get("best", {})
    if best and "retain_err_m" in best:
        betas.append(float(b_tag))
        retain_errs_b.append(best["retain_err_m"])
        fgt_rates_b.append(best.get("lr_test_forget_rate", float("nan")))

fig, ax1 = plt.subplots(figsize=(6, 3.8))
ax2 = ax1.twinx()
ax1.plot(betas, retain_errs_b, "o-", color=C_RETAIN, lw=2, ms=7, label="Retain error (m)")
ax2.plot(betas, fgt_rates_b,   "s--", color=C_FORGET, lw=2, ms=7, label="LR test forget rate")
ax2.axhline(0.50, color="grey", ls=":", lw=1.2, alpha=0.8, label="Target 0.50")
ax1.set_xlabel(r"$\beta$ (retain anchor strength)")
ax1.set_ylabel("Retain mean error (m)", color=C_RETAIN)
ax2.set_ylabel("LR test forget rate", color=C_FORGET)
ax1.tick_params(axis="y", labelcolor=C_RETAIN)
ax2.tick_params(axis="y", labelcolor=C_FORGET)
ax1.set_title(r"$\beta$ ablation: larger $\beta$ → tighter retain → forget rate → 0.50")
lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=9, loc="upper left")
fig.tight_layout()
save("11_beta_anchor_ablation.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
# 12  Error distribution + LR boundary
# ══════════════════════════════════════════════════════════════════════════════
print("[12] error_distribution")
print("  Computing train errors from unlearned model...")
preds_train = get_predictions(model_ul, X_full, device=DEVICE)
errors_train = prediction_errors(preds_train, pos_train)
mia = mia_accuracy(errors_train, y, errors_train, y)
lr_coef = mia["lr_model"].coef_[0, 0]
lr_bias = mia["lr_model"].intercept_[0]
boundary = -lr_bias / lr_coef   # logit = 0 → threshold

fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 2.0, 80)
ax.hist(errors_train[retain_mask], bins=bins, color=C_RETAIN, alpha=0.7,
        density=True, label=f"Retain  (mean {errors_train[retain_mask].mean():.2f} m)")
ax.hist(errors_train[forget_mask], bins=bins, color=C_FORGET, alpha=0.7,
        density=True, label=f"Forget  (mean {errors_train[forget_mask].mean():.2f} m)")
ax.axvline(boundary, color="k", lw=2.0, ls="--",
           label=f"LR boundary {boundary:.3f} m")
ax.set_xlabel("Prediction error (m) vs. original labels")
ax.set_ylabel("Density")
ax.set_title("Per-sample error distribution under the unlearned model\n"
             "Error = corruption magnitude → detector signal")
ax.legend(fontsize=9)
fig.tight_layout()
save("12_error_distribution.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
print(f"\nAll figures saved to {FIG_DIR}")
print(sorted(p.name for p in FIG_DIR.glob("*.pdf")))
