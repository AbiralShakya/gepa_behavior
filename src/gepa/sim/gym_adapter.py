from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union
import importlib
import numpy as np


@dataclass
class GymStepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


def _import_entry_point(entry_point: str) -> Union[type, Callable[..., Any]]:
    if ":" in entry_point:
        module_name, attr_name = entry_point.split(":", 1)
    else:
        # Support dot-path to a callable/class as well
        parts = entry_point.split(".")
        module_name, attr_name = ".".join(parts[:-1]), parts[-1]
    module = importlib.import_module(module_name)
    target = getattr(module, attr_name)
    return target


class GymAdapterEnv:
    """Generic adapter for Gym/Gymnasium-like envs and Diffusion Policy envs.

    - Dynamically imports an entry point (callable or class) to construct the env
    - Normalizes `reset` and `step` to ndarray observations compatible with our runner
    - Supports extracting a key from dict observations via `obs_key`
    """

    def __init__(
        self,
        entry_point: str,
        env_kwargs: Optional[Dict[str, Any]] = None,
        obs_key: Optional[str] = None,
    ) -> None:
        self.entry_point = entry_point
        self.env_kwargs = env_kwargs or {}
        self.obs_key = obs_key
        factory = _import_entry_point(entry_point)
        if callable(factory):
            self.env = factory(**self.env_kwargs)
        else:
            raise RuntimeError(f"Entry point {entry_point} is not callable")

        # Infer action dimension if possible
        self.action_dim: Optional[int] = None
        action_space = getattr(self.env, "action_space", None)
        if action_space is not None and hasattr(action_space, "shape") and action_space.shape is not None:
            self.action_dim = int(action_space.shape[0])
        elif hasattr(self.env, "action_dim"):
            self.action_dim = int(getattr(self.env, "action_dim"))

    def _to_array_obs(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            if self.obs_key is None:
                raise ValueError("Observation is a dict; please set obs_key in config to select a tensor.")
            value = obs[self.obs_key]
            arr = np.array(value, dtype=np.float32)
            return arr
        arr = np.array(obs, dtype=np.float32)
        return arr

    def reset(self) -> np.ndarray:
        result = self.env.reset()
        # Gymnasium returns (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, _ = result
        else:
            obs = result
        return self._to_array_obs(obs)

    def step(self, action: np.ndarray) -> GymStepResult:
        result = self.env.step(action)
        # Gymnasium returns (obs, reward, terminated, truncated, info)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = result  # type: ignore[misc]
        obs_arr = self._to_array_obs(obs)
        return GymStepResult(obs_arr, float(reward), bool(done), info or {})

    def close(self) -> None:
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:
                pass


