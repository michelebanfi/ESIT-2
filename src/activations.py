"""Shared activation-space helpers for probing and unlearning experiments.

ActivationTap:       forward hooks on model.features[i] that clone outputs into .acts.
                     Cloning is required because DichasusPositionPredictor uses inplace ReLU.

fisher_importance:   batch-Fisher diagonal (squared gradients of MSE loss) for a model
                     on a dataset. Pure function: no module-level globals.
"""
import torch
import torch.nn as nn


class ActivationTap:
    """Register forward hooks on model.features[i] for each i in indices.

    .acts[i] holds the last forward-pass output at that layer (cloned).
    Call .clear() between batches and .remove() when done.
    """

    def __init__(self, model, indices):
        self.acts = {}
        self.handles = [
            model.features[i].register_forward_hook(self._make(i))
            for i in indices
        ]

    def _make(self, i):
        def hook(_m, _inp, out):
            self.acts[i] = out.clone()
        return hook

    def clear(self):
        self.acts = {}

    def remove(self):
        for h in self.handles:
            h.remove()


def fisher_importance(model: nn.Module,
                      X: torch.Tensor,
                      Y: torch.Tensor,
                      device: torch.device,
                      batch_size: int = 64) -> dict:
    """Diagonal empirical Fisher for a MSE-loss regression model.

    Returns a dict mapping each parameter name to a same-shaped tensor holding the
    mean of squared gradients accumulated over batches of (X, Y).

    The model runs in .eval() mode so BatchNorm running stats are not updated.
    No side effects on the model's parameter values.
    """
    imp = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    crit = nn.MSELoss()
    model.eval()
    nb = 0
    for i in range(0, len(X), batch_size):
        model.zero_grad(set_to_none=True)
        loss = crit(model(X[i:i + batch_size].to(device)),
                    Y[i:i + batch_size].to(device))
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                imp[n] += p.grad.detach() ** 2
        nb += 1
    model.zero_grad(set_to_none=True)
    return {n: (v / nb).cpu() for n, v in imp.items()}
