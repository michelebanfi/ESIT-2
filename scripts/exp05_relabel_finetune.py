"""Experiment 05 — relabel forget samples with kNN-corrected positions, finetune on full set.

The probe showed the contamination is corrupted position labels: forget samples' CSI
disagrees with their labelled position. So instead of random relabelling, estimate each
forget sample's TRUE position as the mean position of its k nearest retain neighbours
in CSI-PCA space, then finetune on the full train set with corrected labels.

The model then predicts true positions everywhere; its error against the *corrupted*
labels equals the corruption magnitude -> maximal forget/retain separation, while
retain utility is preserved (retain labels are untouched).
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import TensorDataset, DataLoader, random_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import load_model
from src.dataset import format_csi_for_cnn
from src.train import train_epoch, eval_loss

CFG = {
    "lr": 1e-4,
    "epochs": 30,
    "batch_size": 64,
    "val_frac": 0.1,
    "seed": 42,
    "k_neighbours": 10,
}

DATA = ROOT / "data" / "public"
CKPT_IN = ROOT / "data" / "baseline_cnn_task2.pth"
OUT_DIR = ROOT / "experiments" / "exp05_relabel_finetune"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(CFG["seed"])
print(f"Device: {DEVICE}")

print("Loading train data...")
csi_train = np.load(DATA / "task2_train_csi.npy")
pos_train = np.load(DATA / "task2_train_positions.npy")[:, :2].copy()
meta_train = pd.read_csv(DATA / "task2_train_metadata.csv")
forget_mask = meta_train["is_forget"].values == 1
retain_mask = ~forget_mask

# ---- kNN-corrected positions for forget samples ----
print("Correcting forget labels via retain CSI-neighbours...")
n = len(csi_train)
mag = np.abs(csi_train.reshape(n, -1)).astype(np.float32)
pca = PCA(n_components=64, random_state=CFG["seed"])
feats = pca.fit_transform(mag)

nn_idx = (NearestNeighbors(n_neighbors=CFG["k_neighbours"])
          .fit(feats[retain_mask])
          .kneighbors(feats[forget_mask], return_distance=False))
corrected = pos_train[retain_mask][nn_idx].mean(axis=1)
shift = np.linalg.norm(corrected - pos_train[forget_mask], axis=1)
print(f"  mean correction shift = {shift.mean():.3f}m  (median {np.median(shift):.3f}m)")

pos_corrected = pos_train.copy()
pos_corrected[forget_mask] = corrected

X_full = format_csi_for_cnn(csi_train)
Y_full = torch.tensor(pos_corrected, dtype=torch.float32)

# ---- train/val split over the full (corrected) set ----
g = torch.Generator().manual_seed(CFG["seed"])
ds = TensorDataset(X_full, Y_full)
n_val = int(len(ds) * CFG["val_frac"])
train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=g)
train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

model = load_model(str(CKPT_IN), DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
criterion = nn.MSELoss()

best_val = float("inf")
best_path = OUT_DIR / "model_best.pth"

print(f"\nFine-tuning on corrected labels for {CFG['epochs']} epochs...")
for epoch in range(1, CFG["epochs"] + 1):
    t_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    v_loss = eval_loss(model, val_loader, criterion, DEVICE)
    if (epoch % 5 == 0) or epoch == 1:
        print(f"  Epoch {epoch:3d}/{CFG['epochs']}  train_mse={t_loss:.5f}  val_mse={v_loss:.5f}")
    if v_loss < best_val:
        best_val = v_loss
        torch.save(model.state_dict(), best_path)

(OUT_DIR / "config.json").write_text(json.dumps(CFG, indent=2))
print(f"\nBest val_mse={best_val:.5f}  saved -> {best_path}")
print("Run: python scripts/eval_robust.py --ckpt", best_path)
