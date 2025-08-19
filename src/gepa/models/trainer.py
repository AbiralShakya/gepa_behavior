from __future__ import annotations

from typing import Callable, Dict, Iterable
import torch

from .behavior_model import TorchBehaviorModel


class Trainer:
    def __init__(self, model: TorchBehaviorModel) -> None:
        self.model = model

    def train_supervised(self, dataloader: Iterable, max_steps: int = 1000) -> Dict:
        logs = {}
        step = 0
        for batch in dataloader:
            obs, actions = batch
            metrics = self.model.train_step_supervised(obs, actions)
            step += 1
            logs[step] = metrics
            if step >= max_steps:
                break
        return {"steps": step, "final_loss": logs[step]["loss"]}

    def evaluate_policy(self, rollout_fn: Callable[[TorchBehaviorModel], Dict], episodes: int = 5) -> Dict:
        stats = []
        for _ in range(episodes):
            stats.append(rollout_fn(self.model))
        # Aggregate by mean for simplicity
        keys = stats[0].keys()
        return {k: float(torch.tensor([s[k] for s in stats]).float().mean().item()) for k in keys}
