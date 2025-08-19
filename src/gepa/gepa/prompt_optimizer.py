from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import random

from .llm_providers import LLMProvider


@dataclass
class Prompt:
    text: str


@dataclass
class PromptCandidate:
    prompt: Prompt
    scores: Dict[str, float]  # e.g., {"reward": 0.0, "robustness": 0.0}

    def pareto_key(self) -> Tuple:
        # Higher-is-better for each metric
        return tuple(-self.scores[k] for k in sorted(self.scores.keys()))


class GEPAOptimizer:
    def __init__(
        self,
        llm: LLMProvider,
        population_size: int = 8,
        pareto_front_k: int = 4,
        mutation_rate: float = 0.3,
        reflection_weight: float = 0.5,
        evaluate_fn: Callable[[Prompt], Dict[str, float]] | None = None,
    ) -> None:
        self.llm = llm
        self.population_size = population_size
        self.pareto_front_k = pareto_front_k
        self.mutation_rate = mutation_rate
        self.reflection_weight = reflection_weight
        self.evaluate_fn = evaluate_fn or (lambda p: {"reward": 0.0})

    def generate_initial_population(self, base_prompt: str) -> List[Prompt]:
        texts = self.llm.generate(base_prompt, n=self.population_size)
        return [Prompt(t) for t in texts]

    def evaluate_population(self, prompts: List[Prompt]) -> List[PromptCandidate]:
        return [PromptCandidate(p, self.evaluate_fn(p)) for p in prompts]

    def pareto_select(self, candidates: List[PromptCandidate]) -> List[PromptCandidate]:
        # Simple sort by lexicographic on metrics (descending)
        return sorted(candidates, key=lambda c: c.pareto_key())[: self.pareto_front_k]

    def mutate(self, prompt: Prompt) -> Prompt:
        if random.random() > self.mutation_rate:
            return prompt
        # Simple mutation heuristic: append a modifier token
        modifiers = [
            " with smooth motions",
            " prioritizing stability",
            " minimizing energy",
            " with faster convergence",
        ]
        return Prompt(prompt.text + random.choice(modifiers))

    def reflect(self, prompt: Prompt, feedback: Dict[str, float]) -> Prompt:
        # Lightweight reflection: add qualitative hint
        if feedback.get("reward", 0.0) < 0.0:
            return Prompt(prompt.text + " avoid oscillations")
        return prompt

    def step(self, prompts: List[Prompt]) -> List[Prompt]:
        evaluated = self.evaluate_population(prompts)
        front = self.pareto_select(evaluated)
        seeds = [c.prompt for c in front]
        # Generate offspring via mutation and reflection
        offspring: List[Prompt] = []
        for s in seeds:
            mutated = self.mutate(s)
            fb = self.evaluate_fn(mutated)
            reflected = self.reflect(mutated, fb)
            offspring.append(reflected)
        # Fill to population size
        while len(offspring) < self.population_size:
            offspring.append(random.choice(seeds))
        return offspring[: self.population_size]

    def optimize(self, base_prompt: str, iterations: int = 5) -> List[PromptCandidate]:
        population = self.generate_initial_population(base_prompt)
        for _ in range(iterations):
            population = self.step(population)
        final = self.evaluate_population(population)
        return self.pareto_select(final)
