"""Experiment 06 — direct is_forget classifier (offline diagnostic only).

IMPORTANT: This script is an OFFLINE DIAGNOSTIC, not a submission path. It produces
experiments/pseudo_labels/test_proba.npy — a ~96% CV accurate test-set oracle used by
eval_official.py and eval_robust.py to measure Kaggle-accuracy offline, without
leaderboard probing. It does NOT comply with the submission rules (bypasses the CNN).

The contamination probe showed that forget-sample positions disagree with the positions
of their CSI nearest neighbours. This classifier exploits that structure directly.
Features: CSI summary stats (per-antenna power + magnitude), labelled (x,y), and
kNN position-consistency score. Best model (HGB) gets ~96% 5-fold CV accuracy.

Usage:
  python scripts/probing/pseudo_labels.py
Output:
  experiments/pseudo_labels/test_proba.npy   -- P(is_forget=1) for each test sample
  experiments/pseudo_labels/metrics.json
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "public"
OUT_DIR = ROOT / "experiments" / "pseudo_labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
K_NEIGHBOURS = 10

print("Loading data...")
csi_tr = np.load(DATA / "task2_train_csi.npy")
pos_tr = np.load(DATA / "task2_train_positions.npy")[:, :2]
meta_tr = pd.read_csv(DATA / "task2_train_metadata.csv")
y = meta_tr["is_forget"].values

csi_te = np.load(DATA / "task2_test_csi.npy")
pos_te = np.load(ROOT / "data" / "task2_test_positions.npy")[:, :2]
meta_te = pd.read_csv(DATA / "task2_test_metadata.csv")

n_tr, n_te = len(y), len(csi_te)
print(f"  train N={n_tr} (forget rate {y.mean():.4f})   test N={n_te}")


def csi_features(csi: np.ndarray) -> tuple:
    """Return (summary features, |CSI| flattened) for a raw complex CSI array."""
    n = csi.shape[0]
    mag = np.abs(csi.reshape(n, -1)).astype(np.float32)
    ant_power = np.abs(csi.reshape(n, 32, 64)).mean(axis=2)
    summary = np.column_stack([
        ant_power,
        mag.mean(axis=1), mag.std(axis=1),
        mag.max(axis=1), np.median(mag, axis=1),
    ])
    return summary, mag


print("Computing CSI features...")
summary_tr, mag_tr = csi_features(csi_tr)
summary_te, mag_te = csi_features(csi_te)

print("Fitting PCA(64) on train |CSI|...")
pca = PCA(n_components=64, random_state=SEED)
pca_tr = pca.fit_transform(mag_tr)
pca_te = pca.transform(mag_te)

# ---- kNN consistency vs retain reference ----
print("Computing kNN position-consistency (reference = retain train)...")
retain_mask = y == 0
nn = NearestNeighbors(n_neighbors=K_NEIGHBOURS + 1).fit(pca_tr[retain_mask])
ref_pos = pos_tr[retain_mask]


def consistency_features(pca_feats: np.ndarray, labelled_pos: np.ndarray,
                          drop_self: bool) -> np.ndarray:
    """Mean/min distance between labelled position and CSI-neighbours' positions."""
    dist, idx = nn.kneighbors(pca_feats)
    if drop_self:
        # for retain samples the nearest neighbour is themselves (distance 0)
        idx = np.where((dist[:, :1] < 1e-9), idx[:, 1:], idx[:, :-1])
    else:
        idx = idx[:, :K_NEIGHBOURS]
    npos = ref_pos[idx]                                        # (N, k, 2)
    d = np.linalg.norm(npos - labelled_pos[:, None, :], axis=2)
    return np.column_stack([d.mean(axis=1), d.min(axis=1), np.median(d, axis=1)])


cons_tr = consistency_features(pca_tr, pos_tr, drop_self=True)
cons_te = consistency_features(pca_te, pos_te, drop_self=False)

X_tr = np.column_stack([summary_tr, pos_tr, cons_tr])
X_te = np.column_stack([summary_te, pos_te, cons_te])
print(f"  feature matrix: {X_tr.shape}")

# ---- CV evaluation ----
results = {"k_neighbours": K_NEIGHBOURS, "n_features": X_tr.shape[1]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
models = {
    "rf": RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1),
    "hgb": HistGradientBoostingClassifier(random_state=SEED),
}

best_name, best_acc = None, -1.0
for name, clf in models.items():
    proba = cross_val_predict(clf, X_tr, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    acc = accuracy_score(y, proba > 0.5)
    auc = roc_auc_score(y, proba)
    results[name] = {"cv_accuracy": float(acc), "cv_auc": float(auc)}
    print(f"  {name:4s}  CV acc={acc:.4f}  AUC={auc:.4f}")
    if acc > best_acc:
        best_name, best_acc = name, acc

# ---- Fit best model on full train, predict test ----
print(f"\nFitting {best_name} on full train set...")
clf = models[best_name]
clf.fit(X_tr, y)
proba_te = clf.predict_proba(X_te)[:, 1]

pred_05 = (proba_te > 0.5).astype(int)
results["best_model"] = best_name
results["test_forget_rate_at_0.5"] = float(pred_05.mean())
print(f"  predicted test forget rate @0.5 = {pred_05.mean():.4f}")

np.save(OUT_DIR / "test_proba.npy", proba_te)
(OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved -> {OUT_DIR / 'test_proba.npy'}")
print("NOTE: test_proba.npy is an offline diagnostic oracle — never put it in the submission path.")
