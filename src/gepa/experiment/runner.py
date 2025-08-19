from __future__ import annotations

import os
from typing import Optional, List, Dict
import numpy as np
import torch
import typer

from gepa.utils.config import Config, ConfigLoader
from gepa.utils.logging_utils import Logger
from gepa.sim import BulletSimEnv
from gepa.models import TorchBehaviorModel
from gepa.gepa import GEPAOptimizer, Prompt
from gepa.gepa.llm_providers import MockLLM

try:
    from gepa.gepa.llm_providers import OpenAILLM
except Exception:
    OpenAILLM = None

try:
    from gepa.sim.robosuite_env import RoboSuiteEnv
except Exception:
    RoboSuiteEnv = None

from gepa.models.vision import SmallCNN


app = typer.Typer(add_completion=False)


def make_env(cfg: Config):
    if cfg.simulation.backend == "pybullet":
        return BulletSimEnv(
            urdf_path=cfg.simulation.robot_urdf,
            gui=cfg.simulation.gui,
            timestep=cfg.simulation.timestep,
            gravity=cfg.simulation.gravity,
            base_position=cfg.simulation.base_position,
            base_orientation_euler=cfg.simulation.base_orientation_euler,
            max_steps_per_episode=cfg.simulation.max_steps_per_episode,
            enable_camera=cfg.simulation.enable_camera,
            camera_width=cfg.simulation.camera_width,
            camera_height=cfg.simulation.camera_height,
            camera_fov=cfg.simulation.camera_fov,
            camera_distance=cfg.simulation.camera_distance,
            camera_yaw=cfg.simulation.camera_yaw,
            camera_pitch=cfg.simulation.camera_pitch,
        )
    elif cfg.simulation.backend == "robosuite":
        if RoboSuiteEnv is None:
            raise RuntimeError("robosuite backend requested but not installed")
        return RoboSuiteEnv(task="Lift", robots="Panda", horizon=cfg.simulation.max_steps_per_episode, has_renderer=cfg.simulation.gui)
    else:
        raise ValueError(f"Unknown backend {cfg.simulation.backend}")


def make_model(cfg: Config, extra_input_dim: int = 0) -> TorchBehaviorModel:
    return TorchBehaviorModel(
        architecture=cfg.model.architecture,
        input_dim=cfg.model.input_dim,
        action_dim=cfg.model.action_dim,
        hidden_sizes=cfg.model.hidden_sizes,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        device="cpu",
        prompt_conditioning=cfg.model.prompt_conditioning,
        prompt_embed_dim=cfg.model.prompt_embed_dim,
        extra_input_dim=extra_input_dim,
    )


def rollout(env, model: TorchBehaviorModel, steps: int, prompt: Optional[str], vision_encoder: Optional[SmallCNN]) -> Dict[str, float]:
    total_reward = 0.0
    smoothness_penalty = 0.0
    energy_penalty = 0.0
    prev_action = None

    if isinstance(env, BulletSimEnv):
        obs = env.get_observation()
    else:
        obs = env.reset()

    for _ in range(steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        extra = None
        if vision_encoder is not None and isinstance(env, BulletSimEnv):
            rgb = env.render_rgb()
            if rgb is not None:
                img = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                extra = vision_encoder(img)
        action = model.select_action(obs_t, prompt=prompt, extra_features=extra).action.squeeze(0).numpy()
        result = env.step(action)
        obs = result.observation
        total_reward += result.reward

        if prev_action is not None:
            smoothness_penalty += float(np.linalg.norm(action - prev_action))
            energy_penalty += float(np.linalg.norm(action))
        prev_action = action

        if result.done:
            break

    steps_taken = max(1, _ + 1)
    return {
        "reward": float(total_reward),
        "smoothness": -smoothness_penalty / steps_taken,
        "energy": -energy_penalty / steps_taken,
    }


@app.command()
def run(
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
    episodes: Optional[int] = typer.Option(None, help="Override episodes"),
    steps: Optional[int] = typer.Option(None, help="Override steps per episode"),
    robot: Optional[str] = typer.Option(None, help="Override robot URDF path"),
    arch: Optional[str] = typer.Option(None, help="Model architecture override"),
    gepa_iters: int = typer.Option(0, help="Number of GEPA iterations"),
    prompt: Optional[str] = typer.Option(None, help="Manual prompt override"),
    backend: Optional[str] = typer.Option(None, help="Backend override: pybullet|robosuite"),
    camera: bool = typer.Option(False, help="Enable camera (pybullet only)"),
    train_bc_steps: int = typer.Option(0, help="Train supervised BC steps from collected buffer"),
    llm: str = typer.Option("mock", help="LLM provider: mock|openai"),
    openai_model: str = typer.Option("gpt-4o-mini", help="OpenAI model name"),
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
    if backend is not None:
        cfg.simulation.backend = backend
    if camera:
        cfg.simulation.enable_camera = True

    os.makedirs(cfg.experiment.log_dir, exist_ok=True)
    logger = Logger(cfg.experiment.log_dir)

    env = make_env(cfg)
    if isinstance(env, BulletSimEnv):
        obs_dim = env.reset().shape[0]
        cfg.model.input_dim = int(obs_dim)
        cfg.model.action_dim = int(env.num_joints)
    else:
        obs_dim = env.reset().shape[0]
        cfg.model.input_dim = int(obs_dim)
        cfg.model.action_dim = int(env.action_dim)

    extra_input_dim = 128 if (isinstance(env, BulletSimEnv) and cfg.simulation.enable_camera) else 0
    model = make_model(cfg, extra_input_dim=extra_input_dim)

    vision_encoder = SmallCNN(out_dim=extra_input_dim) if extra_input_dim > 0 else None

    active_prompt = prompt or cfg.gepa.base_prompt

    def evaluate_prompt(p: Prompt):
        # Multi-metric for Pareto
        rewards, smooth, energy = 0.0, 0.0, 0.0
        n = max(1, cfg.gepa.evaluation_episodes)
        for _ in range(n):
            _ = env.reset()
            metrics = rollout(env, model, steps=cfg.experiment.steps_per_episode, prompt=p.text, vision_encoder=vision_encoder)
            rewards += metrics["reward"]
            smooth += metrics["smoothness"]
            energy += metrics["energy"]
        return {
            "reward": rewards / n,
            "smoothness": smooth / n,
            "energy": energy / n,
        }

    # Provider selection
    if llm == "openai":
        if OpenAILLM is None:
            raise RuntimeError("OpenAI provider requested but openai package not installed.")
        provider = OpenAILLM(model=openai_model)
    else:
        provider = MockLLM()

    if gepa_iters > 0:
        optimizer = GEPAOptimizer(
            llm=provider,
            population_size=cfg.gepa.population_size,
            pareto_front_k=cfg.gepa.pareto_front_k,
            mutation_rate=cfg.gepa.mutation_rate,
            reflection_weight=cfg.gepa.reflection_weight,
            evaluate_fn=evaluate_prompt,
        )
        best = optimizer.optimize(cfg.gepa.base_prompt, iterations=gepa_iters)
        if len(best) > 0:
            active_prompt = best[0].prompt.text
        for i, cand in enumerate(best):
            logger.log_metrics(i, {
                "gepa_reward": cand.scores.get("reward", 0.0),
                "gepa_smooth": cand.scores.get("smoothness", 0.0),
                "gepa_energy": cand.scores.get("energy", 0.0),
            })

    # Simple in-memory buffer for BC
    buffer_obs: List[np.ndarray] = []
    buffer_act: List[np.ndarray] = []
    buffer_prompts: List[str] = []

    for ep in range(cfg.experiment.episodes):
        _ = env.reset()
        rewards = []
        observations = []
        actions = []
        obs = env.get_observation() if isinstance(env, BulletSimEnv) else env.reset()
        for t in range(cfg.experiment.steps_per_episode):
            observations.append(obs.copy())
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            extra = None
            if vision_encoder is not None and isinstance(env, BulletSimEnv):
                rgb = env.render_rgb()
                if rgb is not None:
                    img = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                    extra = vision_encoder(img)
            action = model.select_action(obs_t, prompt=active_prompt, extra_features=extra).action.squeeze(0).numpy()
            actions.append(action.copy())
            result = env.step(action)
            obs = result.observation
            rewards.append(result.reward)
            # Collect buffer entries
            buffer_obs.append(observations[-1])
            buffer_act.append(actions[-1])
            buffer_prompts.append(active_prompt)
            if result.done:
                break
        logger.log_trajectory(f"episode_{ep}", observations, actions, rewards)
        logger.log_metrics(ep, {"episode_reward": float(sum(rewards)), "prompt_len": len(active_prompt or "")})

    # Optional BC step on collected buffer
    if train_bc_steps > 0 and len(buffer_obs) > 0:
        obs_tensor = torch.tensor(np.array(buffer_obs), dtype=torch.float32)
        act_tensor = torch.tensor(np.array(buffer_act), dtype=torch.float32)
        for step in range(train_bc_steps):
            # Simple minibatch
            idx = np.random.choice(len(buffer_obs), size=min(256, len(buffer_obs)), replace=False)
            mb_obs = obs_tensor[idx]
            mb_act = act_tensor[idx]
            mb_prompts = [buffer_prompts[i] for i in idx]
            metrics = model.train_step_supervised(mb_obs, mb_act, prompts=mb_prompts)
            if (step + 1) % 10 == 0:
                logger.log_metrics(step, {"bc_loss": metrics["loss"]})

    logger.close()
    env.close()


if __name__ == "__main__":
    app()
