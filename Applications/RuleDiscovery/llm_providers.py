"""
LLM Provider Interface

Supports multiple backends:
- MockLLM: For testing without API calls
- AnthropicLLM: Claude API
- OpenAICompatibleLLM: OpenAI-compatible APIs (Argonne, vLLM, TGI, etc.)

Configuration via environment variables:
- LLM_PROVIDER: "mock", "anthropic", "openai_compatible"
- LLM_MODEL: Model name (e.g., "claude-3-haiku-20240307", "meta-llama/Llama-3-70b")
- LLM_API_KEY: API key (for anthropic) or empty for local
- LLM_BASE_URL: Base URL for OpenAI-compatible endpoints
"""

from __future__ import annotations

import os
import random
import re
from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Get completion from LLM."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        pass


class MockLLM(LLMProvider):
    """
    Mock LLM for testing without API calls.
    Generates plausible but simple responses based on patterns.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.call_count = 0

        # Pre-defined experiment suggestions
        self._experiments = [
            "Drop the small red metal sphere into water",
            "Apply electricity to the large blue metal cube",
            "Expose the medium green wood pyramid to fire",
            "Throw the small yellow rubber sphere at a wall",
            "Drop the large red glass cylinder onto the floor",
            "Put the medium blue metal cube in the freezer",
            "Place the small green wood sphere in sunlight",
            "Drop the large yellow glass pyramid into water",
            "Apply electricity to the small red rubber cylinder",
            "Expose the medium blue wood cube to fire",
            "Throw the large green metal sphere at a wall",
            "Drop the small yellow wood pyramid onto the floor",
        ]

        # Simple hypothesis templates
        self._hypothesis_templates = [
            ("Metal objects conduct electricity", 85),
            ("Glass objects shatter when dropped", 80),
            ("Rubber objects bounce when thrown", 75),
            ("Wood objects burn when exposed to fire", 70),
            ("Blue objects float in water", 65),
            ("Large objects sink in water", 60),
            ("Small objects fly far when thrown", 55),
            ("Red objects are fireproof", 50),
        ]

    @property
    def name(self) -> str:
        return "MockLLM"

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Generate mock response based on prompt patterns."""
        self.call_count += 1
        user_lower = user.lower()

        # Detect what kind of response is expected
        if "propose" in user_lower and "experiment" in user_lower:
            return self._generate_experiment()
        elif "hypothes" in user_lower or "rule" in user_lower:
            return self._generate_hypotheses(user)
        elif "evaluate" in user_lower or "compare" in user_lower:
            return self._generate_evaluation(user)
        else:
            return self._generate_generic()

    def _generate_experiment(self) -> str:
        """Generate a random experiment suggestion."""
        return self.rng.choice(self._experiments)

    def _generate_hypotheses(self, context: str) -> str:
        """Generate hypotheses based on context."""
        # Pick 3-5 random hypotheses
        n = self.rng.randint(3, 5)
        selected = self.rng.sample(self._hypothesis_templates, min(n, len(self._hypothesis_templates)))

        lines = []
        for rule, base_conf in selected:
            # Add some randomness to confidence
            conf = max(20, min(95, base_conf + self.rng.randint(-15, 15)))
            lines.append(f"RULE: {rule}")
            lines.append(f"CONFIDENCE: {conf}")
            lines.append("")

        return "\n".join(lines)

    def _generate_evaluation(self, context: str) -> str:
        """Generate evaluation response."""
        score = self.rng.randint(40, 85)
        return f"""RULE_SCORES:
- Metal objects conduct electricity: fully
- Glass objects shatter when dropped: partially
- Rubber objects bounce when thrown: fully
- Wood objects burn in fire: missed
- Blue objects float in water: partially

OVERALL_SCORE: {score}
EXPLANATION: The agents discovered several key rules about material properties,
particularly around conductivity and fragility. Some color-based rules were
partially identified. The fire-related rules were not fully explored.
"""

    def _generate_generic(self) -> str:
        """Generate generic response."""
        return "I observe patterns in the experimental data that suggest certain material and color properties affect outcomes."


class AnthropicLLM(LLMProvider):
    """LLM provider using Anthropic's Claude API."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

    @property
    def name(self) -> str:
        return f"Anthropic({self.model})"

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Get completion from Claude."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        return response.content[0].text


class OpenAICompatibleLLM(LLMProvider):
    """
    LLM provider for OpenAI-compatible APIs.
    Works with vLLM, TGI, Argonne inference, and other compatible endpoints.
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-3-70b-chat-hf",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "not-needed")
        self._client = None

    @property
    def name(self) -> str:
        return f"OpenAI-Compatible({self.model})"

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                )
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Get completion from OpenAI-compatible endpoint."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        return response.choices[0].message.content


def create_llm_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    seed: int = 42,
) -> LLMProvider:
    """
    Factory function to create LLM provider based on configuration.

    Args:
        provider: "mock", "anthropic", or "openai_compatible"
                 Defaults to LLM_PROVIDER env var or "mock"
        model: Model name, defaults to LLM_MODEL env var
        base_url: Base URL for OpenAI-compatible, defaults to LLM_BASE_URL env var
        api_key: API key, defaults to LLM_API_KEY or ANTHROPIC_API_KEY env var
        seed: Random seed for mock provider

    Returns:
        Configured LLMProvider instance
    """
    provider = provider or os.getenv("LLM_PROVIDER", "mock")
    provider = provider.lower().strip()

    if provider == "mock":
        return MockLLM(seed=seed)

    elif provider == "anthropic":
        model = model or os.getenv("LLM_MODEL", "claude-3-haiku-20240307")
        return AnthropicLLM(model=model, api_key=api_key)

    elif provider in ("openai_compatible", "openai", "argonne", "vllm"):
        model = model or os.getenv("LLM_MODEL", "meta-llama/Llama-3-70b-chat-hf")
        return OpenAICompatibleLLM(model=model, base_url=base_url, api_key=api_key)

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            "Use 'mock', 'anthropic', or 'openai_compatible'"
        )


# Convenience function for quick testing
def test_provider(provider: LLMProvider) -> None:
    """Test an LLM provider with a simple prompt."""
    print(f"Testing provider: {provider.name}")

    response = provider.complete(
        system="You are a helpful assistant.",
        user="What is 2+2? Reply with just the number.",
        max_tokens=10
    )
    print(f"Response: {response}")
    print("Provider test passed!\n")


if __name__ == "__main__":
    # Test all providers
    print("=" * 50)
    print("LLM Provider Tests")
    print("=" * 50)

    # Test mock
    print("\n1. Testing MockLLM...")
    mock = MockLLM()
    test_provider(mock)

    # Test Anthropic if key available
    if os.getenv("ANTHROPIC_API_KEY"):
        print("2. Testing AnthropicLLM...")
        anthropic_llm = AnthropicLLM()
        test_provider(anthropic_llm)
    else:
        print("2. Skipping AnthropicLLM (no ANTHROPIC_API_KEY)")

    # Test OpenAI-compatible if URL available
    if os.getenv("LLM_BASE_URL"):
        print("3. Testing OpenAICompatibleLLM...")
        openai_llm = OpenAICompatibleLLM()
        test_provider(openai_llm)
    else:
        print("3. Skipping OpenAICompatibleLLM (no LLM_BASE_URL)")

    print("\nAll available providers tested!")
