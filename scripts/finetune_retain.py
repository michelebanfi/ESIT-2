"""Experiment 01 — finetune baseline checkpoint on the retain set only.

Mirrors the competition notebook's baseline (Naive Unlearning):
  - Load pretrained checkpoint.
  - Fine-tune only on is_forget==0 samples.
  - Evaluate via the MIA pipeline (CNN errors -> LR -> is_forget accuracy).

Good unlearning = higher errors on forget samples -> LR can distinguish them
                  better than in the baseline (higher MIA accuracy = better unlearning signal).
"""
import sys
import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import load_model
from src.dataset import format_csi_for_cnn
from src.metrics import get_predictions, prediction_errors, localization_stats, mia_accuracy
from src.train import train_epoch, eval_loss

# ---- Config ----
CFG = {
    "lr": 1e-4,
    "epochs": 50,
    "batch_size": 64,
    "val_frac": 0.1,
    "seed": 42,
}

DATA = ROOT / "data" / "public"
CKPT_IN = ROOT / "data" / "baseline_cnn_task2.pth"
OUT_DIR = ROOT / "experiments" / "exp01_finetune_retain"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ---- Load data ----
print("Loading train data...")
csi_train = np.load(DATA / "task2_train_csi.npy")
pos_train = np.load(DATA / "task2_train_positions.npy")
meta_train = pd.read_csv(DATA / "task2_train_metadata.csv")

retain_mask = meta_train["is_forget"].values == 0
forget_mask = ~retain_mask
labels = meta_train["is_forget"].values

X_full = format_csi_for_cnn(csi_train)
Y_full = torch.tensor(pos_train[:, :2], dtype=torch.float32)

X_retain = X_full[retain_mask]
Y_retain = Y_full[retain_mask]

print(f"  Retain: {len(X_retain)}  Forget: {forget_mask.sum()}")

# ---- Retain train/val split ----
g = torch.Generator().manual_seed(CFG["seed"])
n_val = int(len(X_retain) * CFG["val_frac"])
n_train = len(X_retain) - n_val
retain_ds = TensorDataset(X_retain, Y_retain)
train_ds, val_ds = random_split(retain_ds, [n_train, n_val], generator=g)

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
print(f"  Train split: {n_train}  Val split: {n_val}")

# ---- Model ----
model = load_model(str(CKPT_IN), DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
criterion = nn.MSELoss()

best_val = float("inf")
best_path = OUT_DIR / "model_best.pth"

# ---- Training ----
print(f"\nFine-tuning for {CFG['epochs']} epochs...")
for epoch in range(1, CFG["epochs"] + 1):
    t_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    v_loss = eval_loss(model, val_loader, criterion, DEVICE)
    if (epoch % 10 == 0) or epoch == 1:
        print(f"  Epoch {epoch:3d}/{CFG['epochs']}  "
              f"train_mse={t_loss:.5f}  val_mse={v_loss:.5f}")
    if v_loss < best_val:
        best_val = v_loss
        torch.save(model.state_dict(), best_path)

print(f"\nLoading best checkpoint (val_mse={best_val:.5f})...")
model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
model.eval()

# ---- Full MIA evaluation ----
print("\nRunning MIA evaluation...")
preds_train = get_predictions(model, X_full, device=DEVICE)
errors_train = prediction_errors(preds_train, Y_full.numpy())
mia = mia_accuracy(errors_train, labels, errors_train, labels)
print(f"  MIA train self-accuracy: {mia['mia_accuracy']*100:.2f}%")

results = {
    "config": CFG,
    "mia_train_self_accuracy": mia["mia_accuracy"],
    "retain": localization_stats(errors_train[retain_mask]),
    "forget": localization_stats(errors_train[forget_mask]),
    "all_train": localization_stats(errors_train),
}

print("\nLocalization errors:")
for split in ("retain", "forget"):
    s = results[split]
    print(f"  {split:7s}  n={s['n_samples']:5d}  "
          f"mean={s['mean_m']:.4f}m  median={s['median_m']:.4f}m")

# Test set
test_pos_path = ROOT / "data" / "task2_test_positions.npy"
csi_test_path = DATA / "task2_test_csi.npy"
if test_pos_path.exists():
    csi_test = np.load(csi_test_path)
    pos_test = np.load(test_pos_path)
    X_test = format_csi_for_cnn(csi_test)
    Y_test = pos_test[:, :2]
    preds_test = get_predictions(model, X_test, device=DEVICE)
    errors_test = prediction_errors(preds_test, Y_test)
    results["test"] = localization_stats(errors_test)
    s = results["test"]
    print(f"  {'test':7s}  n={s['n_samples']:5d}  "
          f"mean={s['mean_m']:.4f}m  median={s['median_m']:.4f}m")

(OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
(OUT_DIR / "config.json").write_text(json.dumps(CFG, indent=2))
print(f"\nSaved -> {OUT_DIR / 'metrics.json'}")

# ---- Update CLAUDE.md changelog ----
claude_md = ROOT / "CLAUDE.md"
baseline_path = ROOT / "experiments" / "exp00_baseline" / "metrics.json"
b = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

today = datetime.date.today().isoformat()
r = results
b_mia = f"{b.get('mia_train_self_accuracy', '?'):.4f}" if b else "?"
b_ret = f"{b.get('retain', {}).get('mean_m', '?'):.4f}" if b else "?"
b_fgt = f"{b.get('forget', {}).get('mean_m', '?'):.4f}" if b else "?"

text = claude_md.read_text()
marker = "<!-- EXPERIMENTS -->"
if marker in text and "| exp01 |" not in text:
    header = ("| id | date | recipe | MIA acc | retain err (m) | forget err (m) | notes |\n"
              "|----|------|--------|---------|---------------|---------------|-------|\n")
    b_row = (f"| exp00 | pretrained | baseline checkpoint "
             f"| {b_mia} | {b_ret} | {b_fgt} | reference — LR trained on train errors |\n")
    e_row = (f"| exp01 | {today} | finetune retain-only "
             f"lr={CFG['lr']} ep={CFG['epochs']} "
             f"| {r['mia_train_self_accuracy']:.4f} "
             f"| {r['retain']['mean_m']:.4f} | {r['forget']['mean_m']:.4f} "
             f"| first unlearning baseline |\n")
    text = text.replace(marker, f"{marker}\n\n{header}{b_row}{e_row}")
    claude_md.write_text(text)
    print(f"Updated {claude_md.name}")
