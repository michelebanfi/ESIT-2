"""Probe the nature of the contamination (is_forget) without the CNN.

Three questions:
  1. Is is_forget predictable from the CSI alone?         (sensor-side contamination)
  2. Is is_forget predictable from the position alone?    (spatial contamination)
  3. Are forget positions inconsistent with their CSI?    (label/position corruption)
     -> kNN in CSI feature space; distance between a sample's labelled position
        and its neighbours' positions. Corrupted labels = large distance.

All classifiers are 5-fold cross-validated. Base rate (always predict retain) = 0.625.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "public"
OUT_DIR = ROOT / "experiments" / "probe_contamination"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

print("Loading data...")
csi = np.load(DATA / "task2_train_csi.npy")           # (N, 4, 2, 4, 64) complex64
pos = np.load(DATA / "task2_train_positions.npy")[:, :2]
meta = pd.read_csv(DATA / "task2_train_metadata.csv")
y = meta["is_forget"].values
n = len(y)
print(f"  N={n}  forget rate={y.mean():.4f}  base acc={(1 - y.mean()):.4f}")

results = {"n": n, "forget_rate": float(y.mean()), "base_accuracy": float(1 - y.mean())}

# ---- Feature sets ----
csi_flat = csi.reshape(n, -1)                          # (N, 2048) complex
mag = np.abs(csi_flat).astype(np.float32)              # magnitudes
ant_power = np.abs(csi.reshape(n, 32, 64)).mean(axis=2)  # per-antenna mean power (N, 32)

summary = np.column_stack([
    ant_power,
    mag.mean(axis=1), mag.std(axis=1),
    mag.max(axis=1), np.median(mag, axis=1),
])
print(f"  summary features: {summary.shape}")

print("Fitting PCA(64) on |CSI|...")
pca = PCA(n_components=64, random_state=SEED)
mag_pca = pca.fit_transform(mag)
print(f"  explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# ---- 1. CSI -> is_forget ----
print("\n[1] CSI -> is_forget (5-fold CV)")
probes = {
    "csi_summary_logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "csi_summary_rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "csi_pca64_logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "csi_pca64_rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
}
feature_map = {
    "csi_summary_logreg": summary, "csi_summary_rf": summary,
    "csi_pca64_logreg": mag_pca, "csi_pca64_rf": mag_pca,
}
for name, clf in probes.items():
    acc = cross_val_score(clf, feature_map[name], y, cv=5, scoring="accuracy", n_jobs=1)
    results[name] = float(acc.mean())
    print(f"  {name:24s} CV acc = {acc.mean():.4f} (+/- {acc.std():.4f})")

# ---- 2. position -> is_forget ----
print("\n[2] position (x, y) -> is_forget (5-fold CV)")
for name, clf in {
    "pos_rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "pos_logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
}.items():
    acc = cross_val_score(clf, pos, y, cv=5, scoring="accuracy", n_jobs=1)
    results[name] = float(acc.mean())
    print(f"  {name:24s} CV acc = {acc.mean():.4f} (+/- {acc.std():.4f})")

# ---- 3. kNN position-consistency in CSI space ----
print("\n[3] kNN position consistency (CSI PCA space, k=10)")
nn = NearestNeighbors(n_neighbors=11).fit(mag_pca)
_, idx = nn.kneighbors(mag_pca)
idx = idx[:, 1:]  # drop self
neighbour_pos = pos[idx]                                # (N, 10, 2)
consistency = np.linalg.norm(neighbour_pos - pos[:, None, :], axis=2).mean(axis=1)

auc = roc_auc_score(y, consistency)
results["knn_consistency_auc"] = float(auc)
print(f"  mean neighbour-distance  retain={consistency[y == 0].mean():.4f}m  "
      f"forget={consistency[y == 1].mean():.4f}m")
print(f"  AUC(is_forget | inconsistency) = {auc:.4f}")

# accuracy if we threshold consistency at the known forget rate
thresh = np.quantile(consistency, 1 - y.mean())
pred = (consistency > thresh).astype(int)
acc = (pred == y).mean()
results["knn_consistency_acc_at_prior"] = float(acc)
print(f"  acc @ prior-quantile threshold = {acc:.4f}")

(OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved -> {OUT_DIR / 'results.json'}")
