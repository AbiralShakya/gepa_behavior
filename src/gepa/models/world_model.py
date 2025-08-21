from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class BaseWorldModel:
    def train_step(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> Dict:
        raise NotImplementedError

    def predict_next(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to(self, device: torch.device) -> "BaseWorldModel":
        raise NotImplementedError


class MLPDynamics(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


@dataclass
class SimpleMLPDynamicsModel(BaseWorldModel):
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device_str: str = "cpu"

    def __post_init__(self):
        self.device = torch.device(self.device_str)
        self.model = MLPDynamics(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.loss_fn = nn.MSELoss()

    def train_step(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> Dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        pred_next = self.model(state.to(self.device), action.to(self.device))
        loss = self.loss_fn(pred_next, next_state.to(self.device))
        loss.backward()
        self.optimizer.step()
        return {"wm_loss": float(loss.detach().cpu().item())}

    def predict_next(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(state.to(self.device), action.to(self.device))

    def rollout_horizon(self, state: torch.Tensor, policy_action_fn, horizon: int = 5) -> torch.Tensor:
        """Generates a simple model rollout embedding by predicting next states under the current policy.

        Returns a concatenated tensor of predicted deltas across horizon for use as features.
        """
        self.model.eval()
        preds = []
        s = state.to(self.device)
        with torch.no_grad():
            for _ in range(max(0, int(horizon))):
                a = policy_action_fn(s)
                ns = self.model(s, a)
                preds.append(ns - s)
                s = ns
        if len(preds) == 0:
            return torch.zeros((state.shape[0], 0), device=state.device)
        return torch.cat(preds, dim=-1)

    def to(self, device: torch.device) -> "SimpleMLPDynamicsModel":
        self.device = device
        self.model.to(device)
        return self


