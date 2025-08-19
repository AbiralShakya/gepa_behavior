from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

try:
    import robosuite as suite
    from robosuite.controllers import load_controller_config
except Exception:  # pragma: no cover
    suite = None


@dataclass
class RSuiteStepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict


class RoboSuiteEnv:
    def __init__(self, task: str = "Lift", robots: str | list[str] = "Panda", horizon: int = 1000, has_renderer: bool = False) -> None:
        if suite is None:
            raise RuntimeError("robosuite is not installed. pip install robosuite")
        self.env = suite.make(
            env_name=task,
            robots=robots,
            has_renderer=has_renderer,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            horizon=horizon,
            control_freq=20,
        )
        self.horizon = horizon
        self.step_count = 0
        obs = self.env.reset()
        self.obs_dim = obs["robot0_proprio-state"].shape[0]
        self.action_dim = self.env.action_dim

    def reset(self) -> np.ndarray:
        self.step_count = 0
        obs = self.env.reset()
        return obs["robot0_proprio-state"].astype(np.float32)

    def step(self, action: np.ndarray) -> RSuiteStepResult:
        obs, reward, done, info = self.env.step(action)
        self.step_count += 1
        o = obs["robot0_proprio-state"].astype(np.float32)
        done = done or (self.step_count >= self.horizon)
        return RSuiteStepResult(o, float(reward), bool(done), info)

    def close(self) -> None:
        self.env.close()
