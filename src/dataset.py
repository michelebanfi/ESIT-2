"""Dataset utilities for ESIT-D2I Task 2.

The raw CSI is complex64 with shape (N, 4, 2, 4, 64).
format_csi_for_cnn mirrors the competition notebook's preprocessing exactly:
  1. Flatten all antenna dims -> (N, 32, 64)
  2. Split real/imag -> (N, 2, 32, 64)
  3. Global standardise (mean/std over the whole batch)
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
