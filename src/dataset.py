"""Dataset utilities for ESIT-D2I Task 2.

format_csi_for_cnn:         mirrors the competition notebook's preprocessing exactly:
                              1. Flatten antenna dims -> (N, 32, 64)
                              2. Split real/imag -> (N, 2, 32, 64)
                              3. Global standardise (one scalar mean/std over the whole array)

knn_corrected_positions:    denoise forget-sample positions via kNN in CSI-PCA space.
                              Forget labels replaced by the mean of their 10 nearest
                              retain-set neighbours' positions (exp05 recipe).

make_tensor_dataset:        convenience loader returning a TensorDataset.
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset


def format_csi_for_cnn(csi_complex: np.ndarray) -> torch.Tensor:
    """Convert complex CSI (N, ..., subcarriers) to float tensor (N, 2, antennas, subcarriers)."""
    n = csi_complex.shape[0]
    subcarriers = csi_complex.shape[-1]
    antennas = int(np.prod(csi_complex.shape[1:-1]))
    csi = csi_complex.reshape(n, antennas, subcarriers)
    csi_2ch = np.stack([np.real(csi), np.imag(csi)], axis=1).astype(np.float32)
    csi_2ch = (csi_2ch - csi_2ch.mean()) / (csi_2ch.std() + 1e-8)
    return torch.tensor(csi_2ch, dtype=torch.float32)


def knn_corrected_positions(csi_complex: np.ndarray,
                            pos: np.ndarray,
                            forget_mask: np.ndarray,
                            k: int = 10,
                            n_pca: int = 64,
                            seed: int = 42) -> np.ndarray:
    """Denoise forget-sample positions via kNN in CSI-PCA space (exp05 recipe).

    For each forget sample, replaces its labelled position with the mean of its k
    nearest retain-set neighbours' positions found in the n_pca-dimensional PCA of
    |CSI| magnitudes.  Retain positions are left untouched.

    Parameters
    ----------
    csi_complex : raw complex CSI array, shape (N, ...).
    pos         : 2-D (x, y) position array, shape (N, 2).
    forget_mask : boolean mask of length N, True = forget sample.
    k, n_pca, seed : hyperparameters (defaults match exp05 / exp08 configs).

    Returns
    -------
    pos_corrected : copy of pos with forget rows replaced by kNN estimates.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    n = len(csi_complex)
    retain_mask = ~forget_mask
    mag = np.abs(csi_complex.reshape(n, -1)).astype(np.float32)
    feats = PCA(n_components=n_pca, random_state=seed).fit_transform(mag)
    nn_idx = (NearestNeighbors(n_neighbors=k)
              .fit(feats[retain_mask])
              .kneighbors(feats[forget_mask], return_distance=False))
    pos_corrected = pos.copy()
    pos_corrected[forget_mask] = pos[retain_mask][nn_idx].mean(axis=1)
    return pos_corrected


def make_tensor_dataset(csi_path: str, pos_path: str,
                        metadata_path: str = None,
                        subset: str = "all") -> TensorDataset:
    """Load and preprocess into a TensorDataset.

    subset: 'retain' | 'forget' | 'all'   (requires metadata_path)
            'test'                         (no metadata needed; pos_path may be None)
    """
    import pandas as pd

    csi_raw = np.load(csi_path)
    pos_raw = np.load(pos_path)

    if metadata_path is not None and subset in ("retain", "forget", "all"):
        meta = pd.read_csv(metadata_path)
        if subset == "retain":
            mask = meta["is_forget"].values == 0
        elif subset == "forget":
            mask = meta["is_forget"].values == 1
        else:
            mask = np.ones(len(meta), dtype=bool)
        csi_raw = csi_raw[mask]
        pos_raw = pos_raw[mask]

    X = format_csi_for_cnn(csi_raw)
    Y = torch.tensor(pos_raw[:, :2], dtype=torch.float32)  # only (x, y)
    return TensorDataset(X, Y)
