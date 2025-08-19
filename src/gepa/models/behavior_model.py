from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .networks import MLPPolicy, RNNPolicy, TransformerPolicy


@dataclass
class ActionOutput:
    action: torch.Tensor
    aux: Dict


class BaseBehaviorModel:
    def select_action(self, observation: torch.Tensor, prompt: Optional[str] = None) -> ActionOutput:
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError


class TorchBehaviorModel(BaseBehaviorModel):
    def __init__(
        self,
        architecture: str,
        input_dim: int,
        action_dim: int,
        hidden_sizes: Iterable[int] = (256, 256),
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.architecture = architecture

        if architecture == "mlp":
            self.policy: nn.Module = MLPPolicy(input_dim, action_dim, hidden_sizes)
        elif architecture == "rnn":
            # Interpret hidden_sizes[0] as hidden size
            hidden_size = int(list(hidden_sizes)[0]) if hidden_sizes else 256
            self.policy = RNNPolicy(input_dim, action_dim, hidden_size=hidden_size)
        elif architecture == "transformer":
            self.policy = TransformerPolicy(input_dim, action_dim)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()

    def select_action(self, observation: torch.Tensor, prompt: Optional[str] = None) -> ActionOutput:
        self.policy.eval()
        with torch.no_grad():
            if self.architecture == "mlp":
                logits = self.policy(observation.to(self.device))
            else:
                # For sequence models, assume observation already has time dim [B, T, D]
                logits = self.policy(observation.to(self.device))
                if isinstance(logits, tuple):
                    logits = logits[0]
        return ActionOutput(action=logits.cpu(), aux={"prompt": prompt})

    def train_step_supervised(self, obs: torch.Tensor, target_actions: torch.Tensor) -> Dict:
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.architecture == "mlp":
            pred = self.policy(obs.to(self.device))
        else:
            pred = self.policy(obs.to(self.device))
            if isinstance(pred, tuple):
                pred = pred[0]
        loss = self.loss_fn(pred, target_actions.to(self.device))
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu().item())}

    def parameters(self):
        return self.policy.parameters()
