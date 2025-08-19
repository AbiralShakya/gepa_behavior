from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import os
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMProvider:
    def generate(self, prompt: str, n: int = 1) -> List[str]:
        raise NotImplementedError


@dataclass
class MockLLM(LLMProvider):
    seed: int = 42

    def generate(self, prompt: str, n: int = 1) -> List[str]:
        return [f"{prompt} [variant {i}]" for i in range(n)]


@dataclass
class OpenAILLM(LLMProvider):
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 256

    def __post_init__(self):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("openai package not installed. pip install openai") from e
        self._OpenAI = OpenAI
        self._client = self._OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _call(self, content: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "Generate concise prompt variants for robot control."}, {"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=1,
        )
        return resp.choices[0].message.content or ""

    def generate(self, prompt: str, n: int = 1) -> List[str]:
        outputs: List[str] = []
        for i in range(n):
            outputs.append(self._call(f"Seed prompt:\n{prompt}\n\nProvide one improved variant focusing on clarity, safety, and stability."))
        return outputs
