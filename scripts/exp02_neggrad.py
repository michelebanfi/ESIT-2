"""Experiment 02 — NegGrad+: retain finetune + clamped gradient ascent on the forget set.

Each step draws a retain batch and a forget batch:
    loss = MSE(retain) - alpha * min(MSE(forget), clamp)
The ascent term is clamped so forget errors are pushed up to ~sqrt(clamp) metres and
no further — unclamped ascent destroys retain utility.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import load_model
from src.dataset import format_csi_for_cnn
from src.train import eval_loss

CFG = {
    "lr": 1e-4,
    "epochs": 15,
    "batch_size": 64,
    "alpha": 0.5,
    "clamp": 0.25,        # MSE clamp ~ (0.5 m)^2 target forget error
    "val_frac": 0.1,
    "seed": 42,
}

DATA = ROOT / "data" / "public"
CKPT_IN = ROOT / "data" / "baseline_cnn_task2.pth"
OUT_DIR = ROOT / "experiments" / "exp02_neggrad"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(CFG["seed"])
print(f"Device: {DEVICE}")

print("Loading train data...")
csi_train = np.load(DATA / "task2_train_csi.npy")
pos_train = np.load(DATA / "task2_train_positions.npy")[:, :2]
meta_train = pd.read_csv(DATA / "task2_train_metadata.csv")
forget_mask = meta_train["is_forget"].values == 1
retain_mask = ~forget_mask

X_full = format_csi_for_cnn(csi_train)
Y_full = torch.tensor(pos_train, dtype=torch.float32)

X_retain, Y_retain = X_full[retain_mask], Y_full[retain_mask]
X_forget, Y_forget = X_full[forget_mask], Y_full[forget_mask]
print(f"  Retain: {len(X_retain)}  Forget: {len(X_forget)}")

# hold out part of retain for checkpoint selection
g = torch.Generator().manual_seed(CFG["seed"])
n_val = int(len(X_retain) * CFG["val_frac"])
perm = torch.randperm(len(X_retain), generator=g)
val_idx, tr_idx = perm[:n_val], perm[n_val:]
val_loader = DataLoader(TensorDataset(X_retain[val_idx], Y_retain[val_idx]),
                        batch_size=256, shuffle=False)
retain_loader = DataLoader(TensorDataset(X_retain[tr_idx], Y_retain[tr_idx]),
                           batch_size=CFG["batch_size"], shuffle=True)
forget_loader = DataLoader(TensorDataset(X_forget, Y_forget),
                           batch_size=CFG["batch_size"], shuffle=True)

model = load_model(str(CKPT_IN), DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
criterion = nn.MSELoss()

best_val = float("inf")
best_path = OUT_DIR / "model_best.pth"

print(f"\nNegGrad+ for {CFG['epochs']} epochs (alpha={CFG['alpha']}, clamp={CFG['clamp']})...")
for epoch in range(1, CFG["epochs"] + 1):
    model.train()
    forget_iter = iter(forget_loader)
    tot_r, tot_f = 0.0, 0.0
    for Xr, Yr in tqdm(retain_loader, leave=False, desc=f"epoch {epoch}"):
        try:
            Xf, Yf = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            Xf, Yf = next(forget_iter)
        Xr, Yr = Xr.to(DEVICE), Yr.to(DEVICE)
        Xf, Yf = Xf.to(DEVICE), Yf.to(DEVICE)

        optimizer.zero_grad()
        loss_r = criterion(model(Xr), Yr)
        loss_f = criterion(model(Xf), Yf)
        loss = loss_r - CFG["alpha"] * torch.clamp(loss_f, max=CFG["clamp"])
        loss.backward()
        optimizer.step()
        tot_r += loss_r.item() * len(Xr)
        tot_f += loss_f.item() * len(Xf)

    v_loss = eval_loss(model, val_loader, criterion, DEVICE)
    print(f"  Epoch {epoch:3d}/{CFG['epochs']}  retain_mse={tot_r / len(tr_idx):.5f}  "
          f"forget_mse={tot_f / len(tr_idx):.5f}  val_mse={v_loss:.5f}")
    if v_loss < best_val:
        best_val = v_loss
        torch.save(model.state_dict(), best_path)

(OUT_DIR / "config.json").write_text(json.dumps(CFG, indent=2))
print(f"\nBest val_mse={best_val:.5f}  saved -> {best_path}")
print("Run: python scripts/eval_robust.py --ckpt", best_path)
