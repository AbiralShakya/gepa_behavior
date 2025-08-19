from __future__ import annotations

from dataclasses import dataclass
from typing import List


class LLMProvider:
    def generate(self, prompt: str, n: int = 1) -> List[str]:
        raise NotImplementedError


@dataclass
class MockLLM(LLMProvider):
    seed: int = 42

    def generate(self, prompt: str, n: int = 1) -> List[str]:
        # Deterministic simple variations for testing
        return [f"{prompt} [variant {i}]" for i in range(n)]
