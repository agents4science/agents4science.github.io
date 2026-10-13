"""
================================================================
llm_fallback.py - LLM provider with automatic fallback chain.

Tries providers in order:
1. OpenAI (if OPENAI_API_KEY is set)
2. Anthropic (if ANTHROPIC_API_KEY is set)
3. Argonne ALCF (if ARGONNE_API_KEY or Globus tokens available)
4. Mock (final fallback)

Dependencies: pip install openai anthropic

Usage:
from llm_fallback import create_llm

llm = create_llm()
response = llm.complete(system="You are helpful.", user="Hello!")
================================================================
"""

import json
import os
from abc import ABC, abstractmethod


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
    """Mock LLM for testing without API calls."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    @property
    def name(self) -> str:
        return "Mock (no API keys found)"

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        return f"[Mock response to: {user[:50]}...]"


class AnthropicLLM(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def name(self) -> str:
        return f"Anthropic ({self.model})"

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


class OpenAICompatibleLLM(LLMProvider):
    """OpenAI-compatible API provider (works with OpenAI, Argonne, vLLM, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = None,
        provider_name: str = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "not-needed")
        self._client = None
        self._provider_name = provider_name

    @property
    def name(self) -> str:
        if self._provider_name:
            return f"{self._provider_name} ({self.model})"
        return f"openai-compatible/{self.model}"

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content


# --- Argonne ALCF helpers ---

ARGONNE_BASE_URL = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
ARGONNE_TOKENS_PATH = os.path.expanduser(
    "~/.globus/app/58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944/inference_app/tokens.json"
)


def _load_globus_token(token_path: str) -> str | None:
    """Load access token from Globus tokens.json file."""
    try:
        with open(token_path) as f:
            tokens = json.load(f)
        # Token is nested under the gateway client ID
        gateway_id = "681c10cc-f684-4540-bcd7-0b4df3bc26ef"

        # Handle new Globus SDK format (v2.0): data.DEFAULT.<resource_server>
        if "data" in tokens:
            data = tokens.get("data", {}).get("DEFAULT", {})
            return data.get(gateway_id, {}).get("access_token")

        # Handle old format: <resource_server>.access_token
        return tokens.get(gateway_id, {}).get("access_token")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def _get_argonne_api_key() -> str | None:
    """Get Argonne API key from env var or Globus tokens file."""
    api_key = os.getenv("ARGONNE_API_KEY") or os.getenv("LLM_API_KEY")

    if api_key:
        # Clean up any newlines or whitespace
        api_key = api_key.strip().split('\n')[0].strip()

        # Check if it's a path to tokens.json
        if "tokens.json" in api_key and os.path.isfile(api_key):
            return _load_globus_token(api_key)

        # If it looks like a valid token (no path separators), use it directly
        if "/" not in api_key and len(api_key) > 20:
            return api_key

    # Fall back to default tokens path
    if os.path.isfile(ARGONNE_TOKENS_PATH):
        return _load_globus_token(ARGONNE_TOKENS_PATH)

    return None


# --- Factory function ---

def create_llm(model: str = None, verbose: bool = True) -> LLMProvider:
    """
    Create an LLM provider with fallback chain:
    1. OpenAI (if OPENAI_API_KEY is set)
    2. Anthropic (if ANTHROPIC_API_KEY is set)
    3. Argonne ALCF (if ARGONNE_API_KEY or Globus tokens available)
    4. Mock (final fallback)
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    argonne_key = _get_argonne_api_key()

    if openai_key:
        openai_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if verbose:
            print(f"[LLM] Using OpenAI ({openai_model})")
        return OpenAICompatibleLLM(
            model=openai_model,
            base_url="https://api.openai.com/v1",
            api_key=openai_key,
            provider_name="OpenAI",
        )
    elif anthropic_key:
        anthropic_model = model or "claude-3-haiku-20240307"
        if verbose:
            print(f"[LLM] Using Anthropic ({anthropic_model})")
        return AnthropicLLM(
            model=anthropic_model,
            api_key=anthropic_key,
        )
    elif argonne_key:
        argonne_model = model or "meta-llama/Meta-Llama-3.1-70B-Instruct"
        if verbose:
            print(f"[LLM] Using Argonne ALCF ({argonne_model})")
        return OpenAICompatibleLLM(
            model=argonne_model,
            base_url=ARGONNE_BASE_URL,
            api_key=argonne_key,
            provider_name="Argonne Inference Service",
        )
    else:
        if verbose:
            print("[LLM] Using Mock (no API keys found)")
        return MockLLM(seed=42)


if __name__ == "__main__":
    llm = create_llm()
    response = llm.complete(
        system="You are a helpful assistant.",
        user="What is 2 + 2?",
        max_tokens=100,
    )
    print(f"Response from {llm.name}:\n{response}")
