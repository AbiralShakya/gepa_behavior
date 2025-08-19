from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 3, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, H, W]
        h = self.net(x)
        h = h.flatten(1)
        return self.proj(h)
