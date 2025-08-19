from __future__ import annotations

from typing import Iterable, List
import matplotlib.pyplot as plt


def plot_trajectory(rewards: Iterable[float]):
    r = list(rewards)
    plt.figure()
    plt.plot(r)
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Episode Reward Trajectory")
    plt.tight_layout()


def plot_metric_series(steps: Iterable[int], values: Iterable[float], name: str):
    s = list(steps)
    v = list(values)
    plt.figure()
    plt.plot(s, v)
    plt.xlabel("Step")
    plt.ylabel(name)
    plt.title(name)
    plt.tight_layout()
