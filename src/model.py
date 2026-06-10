"""DichasusPositionPredictor — exact architecture from competition_notebook.ipynb.

Input:  (B, 2, Antennas, Subcarriers)  — real + imaginary channels
Output: (B, 2)                          — predicted (x, y) coordinates
"""
import torch
import torch.nn as nn


class DichasusPositionPredictor(nn.Module):
    def __init__(self, in_channels=2, output_dim=2):
        super().__init__()
        c1, c2, c3, c4, c5, c6 = 32, 64, 128, 256, 512, 512

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),

            nn.Conv2d(c5, c6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c6),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c6, output_dim),
        )

    def forward(self, x):
        return self.regression_head(self.features(x))


def load_model(checkpoint_path: str, device: torch.device) -> DichasusPositionPredictor:
    model = DichasusPositionPredictor().to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model
