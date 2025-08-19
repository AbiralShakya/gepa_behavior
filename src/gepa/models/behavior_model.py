from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim

from .networks import MLPPolicy, RNNPolicy, TransformerPolicy
from .conditioning import TextConditioner


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
        prompt_conditioning: bool = False,
        prompt_embed_dim: int = 128,
    ) -> None:
        self.device = torch.device(device)
        self.architecture = architecture
        self.prompt_conditioning = prompt_conditioning
        self.prompt_embed_dim = prompt_embed_dim

        effective_input_dim = input_dim + (prompt_embed_dim if prompt_conditioning and architecture == "mlp" else 0)

        if architecture == "mlp":
            self.policy: nn.Module = MLPPolicy(effective_input_dim, action_dim, hidden_sizes)
        elif architecture == "rnn":
            hidden_size = int(list(hidden_sizes)[0]) if hidden_sizes else 256
            self.policy = RNNPolicy(input_dim, action_dim, hidden_size=hidden_size)
        elif architecture == "transformer":
            self.policy = TransformerPolicy(input_dim, action_dim)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()

        self.conditioner = TextConditioner(embedding_dim=prompt_embed_dim).to(self.device) if prompt_conditioning else None

    def _cond_embed(self, prompts: List[str]) -> Optional[torch.Tensor]:
        if self.conditioner is None:
            return None
        self.conditioner.eval()
        with torch.no_grad():
            return self.conditioner(prompts)

    def select_action(self, observation: torch.Tensor, prompt: Optional[str] = None) -> ActionOutput:
        self.policy.eval()
        with torch.no_grad():
            if self.architecture == "mlp":
                x = observation.to(self.device)
                if self.conditioner is not None and prompt is not None:
                    emb = self._cond_embed([prompt])  # [1, D]
                    x = torch.cat([x, emb], dim=-1)
                logits = self.policy(x)
            else:
                logits = self.policy(observation.to(self.device))
                if isinstance(logits, tuple):
                    logits = logits[0]
        return ActionOutput(action=logits.cpu(), aux={"prompt": prompt})

    def train_step_supervised(self, obs: torch.Tensor, target_actions: torch.Tensor, prompts: Optional[List[str]] = None) -> Dict:
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.architecture == "mlp":
            x = obs.to(self.device)
            if self.conditioner is not None and prompts is not None:
                emb = self.conditioner(prompts)
                x = torch.cat([x, emb], dim=-1)
            pred = self.policy(x)
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
