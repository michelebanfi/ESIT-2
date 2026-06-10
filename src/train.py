import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for X, Y in tqdm(loader, leave=False, desc="train"):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader,
              criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        total_loss += criterion(model(X), Y).item() * len(X)
    return total_loss / len(loader.dataset)
