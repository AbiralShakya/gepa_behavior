from __future__ import annotations

from typing import Iterable, List
import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden_sizes: Iterable[int], out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = in_dim
    for hs in hidden_sizes:
        layers += [nn.Linear(last, hs), nn.ReLU()]
        last = hs
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class MLPPolicy(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: Iterable[int]) -> None:
        super().__init__()
        self.net = _mlp(input_dim, hidden_sizes, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, D]
        return self.net(x)


class RNNPolicy(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int = 256, num_layers: int = 1) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, action_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):  # x: [B, T, D]
        out, h_n = self.rnn(x, h)
        logits = self.head(out[:, -1, :])
        return logits, h_n


class TransformerPolicy(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, D]
        z = self.input_proj(x)
        h = self.encoder(z)
        return self.head(h[:, -1, :])
