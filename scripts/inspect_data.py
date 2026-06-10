"""Sanity-check the downloaded data: shapes, NaN counts, is_forget distribution, position bounds."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "public"

if not DATA.exists():
    print("Run scripts/download_data.py first.")
    sys.exit(1)

train_csi = np.load(DATA / "task2_train_csi.npy")
train_pos = np.load(DATA / "task2_train_positions.npy")
train_meta = pd.read_csv(DATA / "task2_train_metadata.csv")

print("=== Train CSI (raw complex) ===")
print(f"  shape : {train_csi.shape}  dtype={train_csi.dtype}")
print(f"  NaNs  : {np.isnan(train_csi).sum()}")

print("\n=== Train Positions ===")
print(f"  shape : {train_pos.shape}  dtype={train_pos.dtype}")
for i, ax in enumerate("xyz"):
    print(f"  {ax}: [{train_pos[:, i].min():.2f}, {train_pos[:, i].max():.2f}]")

print("\n=== Train Metadata ===")
vc = train_meta["is_forget"].value_counts().sort_index()
print(f"  is_forget distribution:\n{vc.to_string()}")
print(f"  n_retain (is_forget=0): {(train_meta['is_forget']==0).sum()}")
print(f"  n_forget (is_forget=1): {(train_meta['is_forget']==1).sum()}")
print(f"  unique sequences: {train_meta['sequence_id'].nunique()}")

test_csi = np.load(DATA / "task2_test_csi.npy")
test_pos_path = ROOT / "data" / "task2_test_positions.npy"
test_meta = pd.read_csv(DATA / "task2_test_metadata.csv")

print("\n=== Test CSI ===")
print(f"  shape : {test_csi.shape}  dtype={test_csi.dtype}")

if test_pos_path.exists():
    test_pos = np.load(test_pos_path)
    print(f"\n=== Test Positions ===  shape: {test_pos.shape}")

print("\n=== Test Metadata ===")
print(f"  columns: {list(test_meta.columns)}")
print(test_meta.head().to_string())
