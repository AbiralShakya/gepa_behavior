from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, List

import importlib
import torch

from .behavior_model import BaseBehaviorModel, ActionOutput


def _import_symbol(path: str):
    if ":" in path:
        mod, sym = path.split(":", 1)
    else:
        parts = path.split(".")
        mod, sym = ".".join(parts[:-1]), parts[-1]
    module = importlib.import_module(mod)
    return getattr(module, sym)


@dataclass
class DiffusionPolicyAdapter(BaseBehaviorModel):
    policy_ctor_path: str
    ctor_kwargs: Dict[str, Any]
    device: str = "cpu"

    def __post_init__(self):
        ctor = _import_symbol(self.policy_ctor_path)
        self.policy = ctor(**(self.ctor_kwargs or {}))
        self.device_t = torch.device(self.device)
        if hasattr(self.policy, "to"):
            self.policy.to(self.device_t)
        self.policy.eval()

    def select_action(self, observation: torch.Tensor, prompt: Optional[str] = None, extra_features: Optional[torch.Tensor] = None) -> ActionOutput:
        self.policy.eval()
        with torch.no_grad():
            # Expect DP-style .predict_action(obs_dict) or forward
            if isinstance(observation, torch.Tensor):
                obs_t = observation.to(self.device_t)
            else:
                obs_t = observation
            if hasattr(self.policy, "predict_action"):
                act = self.policy.predict_action({"state": obs_t})
            else:
                act = self.policy(obs_t)
        if isinstance(act, tuple):
            act = act[0]
        return ActionOutput(action=act.detach().cpu(), aux={"prompt": prompt})

    def parameters(self):
        # Often frozen during evaluation; expose if needed
        if hasattr(self.policy, "parameters"):
            return self.policy.parameters()
        return []


