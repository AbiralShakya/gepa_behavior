from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import yaml
import copy


@dataclass
class SimulationConfig:
    timestep: float = 1.0 / 240.0
    gravity: float = -9.81
    gui: bool = False
    max_steps_per_episode: int = 1000
    robot_urdf: str = "kuka_iiwa/model.urdf"
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_orientation_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class ModelConfig:
    architecture: str = "mlp"  # mlp | rnn | transformer
    input_dim: int = 32
    action_dim: int = 7
    hidden_sizes: tuple[int, ...] = (256, 256)
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    gamma: float = 0.99
    seed: int = 42


@dataclass
class GEPAConfig:
    llm_provider: str = "mock"  # mock | openai | anthropic | vertex_ai
    population_size: int = 8
    pareto_front_k: int = 4
    mutation_rate: float = 0.3
    reflection_weight: float = 0.5
    evaluation_episodes: int = 3
    max_iterations: int = 10
    base_prompt: str = (
        "You are optimizing a robot control prompt to achieve the task described by the user."
    )


@dataclass
class ExperimentConfig:
    project_name: str = "gepa_behavior"
    log_dir: str = "./runs"
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42
    episodes: int = 10
    steps_per_episode: int = 500


@dataclass
class Config:
    simulation: SimulationConfig = SimulationConfig()
    model: ModelConfig = ModelConfig()
    gepa: GEPAConfig = GEPAConfig()
    experiment: ExperimentConfig = ExperimentConfig()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfigLoader:
    """Loads YAML configs and supports in-memory overrides."""

    @staticmethod
    def from_yaml(path: str) -> Config:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return ConfigLoader.from_dict(raw)

    @staticmethod
    def from_dict(cfg_dict: Dict[str, Any]) -> Config:
        # Deep copy to avoid side effects
        data = copy.deepcopy(cfg_dict)

        def update_dataclass(dc, updates: Optional[Dict[str, Any]]):
            if not updates:
                return dc
            for key, value in updates.items():
                if hasattr(dc, key):
                    setattr(dc, key, value)
            return dc

        cfg = Config()
        update_dataclass(cfg.simulation, data.get("simulation"))
        update_dataclass(cfg.model, data.get("model"))
        update_dataclass(cfg.gepa, data.get("gepa"))
        update_dataclass(cfg.experiment, data.get("experiment"))
        return cfg

    @staticmethod
    def merge(base: Config, overrides: Optional[Dict[str, Any]]) -> Config:
        if not overrides:
            return base
        base_dict = base.to_dict()
        merged = copy.deepcopy(base_dict)

        def recursive_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    d[k] = recursive_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        merged = recursive_update(merged, overrides)
        return ConfigLoader.from_dict(merged)
