"""Evaluation metrics for Task 2.

The Kaggle submission asks to predict `is_forget` for each test sample.
The pipeline:
  1. CNN predicts (x, y) positions.
  2. Compute Euclidean prediction error vs. true positions.
  3. Train a LogisticRegression on train errors (label = is_forget) to learn the threshold.
  4. Apply to test errors -> is_forget predictions -> submit.

Higher error on forget samples = better unlearning signal for the LR.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def get_predictions(model: torch.nn.Module,
                    X: torch.Tensor,
                    batch_size: int = 256,
                    device: torch.device = torch.device("cpu")) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].to(device)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds, axis=0)


def prediction_errors(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """2D Euclidean error per sample."""
    return np.linalg.norm(preds - targets, axis=1)


def localization_stats(errors: np.ndarray) -> dict:
    return {
        "mean_m": float(errors.mean()),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
        "n_samples": int(len(errors)),
    }


def mia_accuracy(train_errors: np.ndarray, train_labels: np.ndarray,
                 eval_errors: np.ndarray, eval_labels: np.ndarray) -> dict:
    """Train LogisticRegression on train errors, evaluate on eval split."""
    lr = LogisticRegression(max_iter=1000)
    lr.fit(train_errors.reshape(-1, 1), train_labels)
    preds = lr.predict(eval_errors.reshape(-1, 1))
    acc = accuracy_score(eval_labels, preds)
    return {"mia_accuracy": float(acc), "lr_model": lr}


def gmm_threshold_predictions(errors: np.ndarray, seed: int = 42) -> dict:
    """Unsupervised is_forget labels: 2-component GMM on log-errors.

    Avoids transferring an absolute error threshold from the (memorised) train
    distribution to the test distribution — the split is found within the set itself.
    The high-error component is labelled forget.
    """
    from sklearn.mixture import GaussianMixture

    log_err = np.log(errors + 1e-9).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=seed, n_init=5)
    comp = gmm.fit_predict(log_err)
    forget_comp = int(np.argmax(gmm.means_.ravel()))
    preds = (comp == forget_comp).astype(int)
    return {
        "predictions": preds,
        "forget_rate": float(preds.mean()),
        "component_means_m": np.exp(gmm.means_.ravel()).tolist(),
    }
