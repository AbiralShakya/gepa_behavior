from __future__ import annotations

import os
from typing import Optional
import numpy as np
import torch
import typer

from gepa.utils.config import Config, ConfigLoader
from gepa.utils.logging_utils import Logger
from gepa.sim import BulletSimEnv
from gepa.models import TorchBehaviorModel
from gepa.gepa import GEPAOptimizer, MockLLM, Prompt


app = typer.Typer(add_completion=False)


def make_env(cfg: Config) -> BulletSimEnv:
    return BulletSimEnv(
        urdf_path=cfg.simulation.robot_urdf,
        gui=cfg.simulation.gui,
        timestep=cfg.simulation.timestep,
        gravity=cfg.simulation.gravity,
        base_position=cfg.simulation.base_position,
        base_orientation_euler=cfg.simulation.base_orientation_euler,
        max_steps_per_episode=cfg.simulation.max_steps_per_episode,
    )


def make_model(cfg: Config) -> TorchBehaviorModel:
    return TorchBehaviorModel(
        architecture=cfg.model.architecture,
        input_dim=cfg.model.input_dim,
        action_dim=cfg.model.action_dim,
        hidden_sizes=cfg.model.hidden_sizes,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        device="cpu",
    )


def rollout(env: BulletSimEnv, model: TorchBehaviorModel, steps: int) -> float:
    obs = env.get_observation()
    total_reward = 0.0
    for _ in range(steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = model.select_action(obs_t).action.squeeze(0).numpy()
        result = env.step(action)
        obs = result.observation
        total_reward += result.reward
        if result.done:
            break
    return float(total_reward)


@app.command()
def run(
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
    episodes: Optional[int] = typer.Option(None, help="Override episodes"),
    steps: Optional[int] = typer.Option(None, help="Override steps per episode"),
    robot: Optional[str] = typer.Option(None, help="Override robot URDF path"),
    arch: Optional[str] = typer.Option(None, help="Model architecture override"),
    gepa_iters: int = typer.Option(0, help="Number of GEPA iterations"),
):
    base_cfg = Config()
    cfg = ConfigLoader.merge(base_cfg, ConfigLoader.from_yaml(config).to_dict() if config else None)
    if episodes is not None:
        cfg.experiment.episodes = episodes
    if steps is not None:
        cfg.experiment.steps_per_episode = steps
    if robot is not None:
        cfg.simulation.robot_urdf = robot
    if arch is not None:
        cfg.model.architecture = arch

    os.makedirs(cfg.experiment.log_dir, exist_ok=True)
    logger = Logger(cfg.experiment.log_dir)

    env = make_env(cfg)
    obs_dim = env.reset().shape[0]
    cfg.model.input_dim = int(obs_dim)
    cfg.model.action_dim = int(env.num_joints)
    model = make_model(cfg)

    # Simple GEPA evaluation function: average reward over N episodes
    def evaluate_prompt(p: Prompt):
        total = 0.0
        for _ in range(max(1, cfg.gepa.evaluation_episodes)):
            env.reset()
            r = rollout(env, model, steps=cfg.experiment.steps_per_episode)
            total += r
        return {"reward": total / max(1, cfg.gepa.evaluation_episodes)}

    if gepa_iters > 0:
        optimizer = GEPAOptimizer(
            llm=MockLLM(),
            population_size=cfg.gepa.population_size,
            pareto_front_k=cfg.gepa.pareto_front_k,
            mutation_rate=cfg.gepa.mutation_rate,
            reflection_weight=cfg.gepa.reflection_weight,
            evaluate_fn=evaluate_prompt,
        )
        best = optimizer.optimize(cfg.gepa.base_prompt, iterations=gepa_iters)
        for i, cand in enumerate(best):
            logger.log_metrics(i, {f"gepa_reward": cand.scores.get("reward", 0.0)})

    for ep in range(cfg.experiment.episodes):
        obs = env.reset()
        rewards = []
        observations = []
        actions = []
        for t in range(cfg.experiment.steps_per_episode):
            observations.append(obs.copy())
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = model.select_action(obs_t).action.squeeze(0).numpy()
            actions.append(action.copy())
            result = env.step(action)
            obs = result.observation
            rewards.append(result.reward)
            if result.done:
                break
        logger.log_trajectory(f"episode_{ep}", observations, actions, rewards)
        logger.log_metrics(ep, {"episode_reward": float(sum(rewards))})

    logger.close()
    env.close()


if __name__ == "__main__":
    app()
