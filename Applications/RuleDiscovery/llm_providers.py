"""
LLM Provider Interface

Supports multiple backends:
- MockLLM: For testing without API calls
- AnthropicLLM: Claude API
- OpenAICompatibleLLM: OpenAI-compatible APIs (Argonne, vLLM, TGI, etc.)
- EmbeddedLLM: Local model on GPU (for HPC: Polaris, Aurora)

Configuration via environment variables:
- LLM_PROVIDER: "mock", "anthropic", "openai_compatible", "embedded"
- LLM_MODEL: Model name (e.g., "claude-3-haiku-20240307", "meta-llama/Llama-3-70b")
- LLM_API_KEY: API key (for anthropic) or empty for local
- LLM_BASE_URL: Base URL for OpenAI-compatible endpoints
- LLM_DEVICE: GPU device for embedded (e.g., "cuda:0", "xpu:0")
- LLM_BACKEND: Backend for embedded ("vllm" or "transformers")

For HPC usage (Polaris/Aurora), use create_embedded_llms() to create
one LLM instance per agent, each on a different GPU:

    from llm_providers import create_embedded_llms

    # Polaris: 4 A100 GPUs per node
    llms = create_embedded_llms(
        model="meta-llama/Llama-3.1-8B-Instruct",
        n_agents=4,
        backend="vllm",
        base_device="cuda",
        gpus_per_node=4,
    )

    # Aurora: 6 Intel GPUs per node
    llms = create_embedded_llms(
        model="meta-llama/Llama-3.1-8B-Instruct",
        n_agents=6,
        backend="transformers",  # vLLM may not support Intel yet
        base_device="xpu",
        gpus_per_node=6,
    )
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


def _load_globus_token(path: str) -> str:
    """Load access token from Globus tokens.json file."""
    import json
    with open(path) as f:
        data = json.load(f)
    # Look for the inference API token (resource server 681c10cc-f684-4540-bcd7-0b4df3bc26ef)
    tokens = data.get("data", {}).get("DEFAULT", {})
    # Try inference API resource server first
    for key, value in tokens.items():
        if key == "681c10cc-f684-4540-bcd7-0b4df3bc26ef":
            return value["access_token"]
    # Fall back to any non-auth.globus.org token
    for key, value in tokens.items():
        if key != "auth.globus.org":
            return value["access_token"]
    raise ValueError(f"No inference token found in {path}")


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
        n_agents: int = 1,
    ):
        self.model = model
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
        api_key = api_key or os.getenv("LLM_API_KEY", "not-needed")
        # If api_key is a path to a Globus tokens.json file, load the token
        if api_key.endswith("tokens.json") and os.path.isfile(api_key):
            api_key = _load_globus_token(api_key)
        self.api_key = api_key
        self.n_agents = n_agents  # Used to scale backoff
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

    def _is_gpt_oss(self) -> bool:
        """Check if this is a gpt-oss model (requires special handling)."""
        return "gpt-oss" in self.model.lower()

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Get completion from OpenAI-compatible endpoint with retry on rate limit."""
        import time as _time
        import math

        max_retries = 6
        # Scale base delay with number of agents: 1 agent = 1s, 128 agents = 8s
        agent_scale = 1 + math.log2(max(1, self.n_agents))  # 1->1, 2->2, 4->3, 8->4, 16->5, 32->6, 64->7, 128->8
        base_delay = agent_scale

        # gpt-oss models: merge system into user message, require temperature>0,
        # and need more max_tokens because they use tokens for internal reasoning
        if self._is_gpt_oss():
            messages = [
                {"role": "user", "content": f"{system}\n\n{user}"},
            ]
            temperature = 0.7
            # gpt-oss needs extra tokens for reasoning - multiply by 5, minimum 500
            max_tokens = max(500, max_tokens * 5)
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            temperature = None  # Use model default

        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": messages,
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                return content if content is not None else ""

            except Exception as e:
                error_str = str(e)
                # Check for rate limit (429) or overloaded errors
                if "429" in error_str or "rate" in error_str.lower() or "too many" in error_str.lower():
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"    [Rate limit hit, {self.n_agents} agents, waiting {delay:.1f}s before retry {attempt + 2}/{max_retries}]")
                        _time.sleep(delay)
                        continue
                # Re-raise if not a rate limit error or out of retries
                raise


class EmbeddedLLM(LLMProvider):
    """
    Embedded LLM for running local models on HPC (Polaris, Aurora).

    Each agent gets its own model instance on a specific GPU.
    Supports vLLM (preferred) or HuggingFace transformers (fallback).

    Usage on Polaris (A100 GPUs):
        llm = EmbeddedLLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            device="cuda:0",  # or "cuda:1", etc.
            backend="vllm",   # or "transformers"
        )

    Usage on Aurora (Intel GPUs):
        llm = EmbeddedLLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            device="xpu:0",
            backend="transformers",  # vLLM may not support Intel yet
        )
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda:0",
        backend: str = "vllm",
        max_model_len: int = 4096,
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        self.model_name = model
        self.device = device
        self.backend = backend.lower()
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code

        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return f"Embedded({self.model_name.split('/')[-1]}, {self.device})"

    def _init_vllm(self):
        """Initialize vLLM backend."""
        try:
            from vllm import LLM, SamplingParams
            import torch

            # Parse device index
            if "cuda:" in self.device:
                gpu_id = int(self.device.split(":")[-1])
                # Set CUDA_VISIBLE_DEVICES to restrict to this GPU
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            elif self.device == "cuda":
                pass  # Use default GPU
            else:
                raise ValueError(f"vLLM requires CUDA device, got: {self.device}")

            self._model = LLM(
                model=self.model_name,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
                enforce_eager=True,  # Required for some GPUs
            )
            self._sampling_params_class = SamplingParams

        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm\n"
                "Or use backend='transformers' as fallback."
            )

    def _init_transformers(self):
        """Initialize HuggingFace transformers backend."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Determine torch dtype
            if self.dtype == "auto":
                torch_dtype = torch.float16
            elif self.dtype == "float16":
                torch_dtype = torch.float16
            elif self.dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            elif self.dtype == "float32":
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.float16

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.device if ":" in self.device else "auto",
                trust_remote_code=self.trust_remote_code,
            )

            # Set pad token if not set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

        except ImportError:
            raise ImportError(
                "transformers not installed. Install with: pip install transformers torch"
            )

    def _ensure_initialized(self):
        """Lazy initialization of model."""
        if self._model is not None:
            return

        if self.backend == "vllm":
            self._init_vllm()
        elif self.backend in ("transformers", "hf", "huggingface"):
            self._init_transformers()
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Use 'vllm' or 'transformers'.")

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Get completion from embedded model."""
        self._ensure_initialized()

        # Format prompt (Llama-style chat template)
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        if self.backend == "vllm":
            return self._complete_vllm(prompt, max_tokens)
        else:
            return self._complete_transformers(prompt, max_tokens)

    def _complete_vllm(self, prompt: str, max_tokens: int) -> str:
        """Generate with vLLM."""
        sampling_params = self._sampling_params_class(
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        outputs = self._model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _complete_transformers(self, prompt: str, max_tokens: int) -> str:
        """Generate with transformers."""
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only the generated part (exclude input)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


def create_embedded_llms(
    model: str,
    n_agents: int,
    backend: str = "vllm",
    base_device: str = "cuda",
    gpus_per_node: int = 4,
    **kwargs,
) -> list[EmbeddedLLM]:
    """
    Create multiple EmbeddedLLM instances, one per agent, assigned to different GPUs.

    This is designed for HPC systems like Polaris (4 A100 GPUs/node) or
    Aurora (6 Intel GPUs/node).

    Args:
        model: HuggingFace model name
        n_agents: Number of agents (each gets one LLM)
        backend: "vllm" or "transformers"
        base_device: "cuda" for NVIDIA, "xpu" for Intel
        gpus_per_node: Number of GPUs per node (4 for Polaris, 6 for Aurora)
        **kwargs: Additional args passed to EmbeddedLLM

    Returns:
        List of EmbeddedLLM instances

    Example:
        # Create 4 agents on Polaris, each on a different A100
        llms = create_embedded_llms(
            model="meta-llama/Llama-3.1-8B-Instruct",
            n_agents=4,
            backend="vllm",
            base_device="cuda",
            gpus_per_node=4,
        )
        # llms[0] on cuda:0, llms[1] on cuda:1, etc.
    """
    llms = []
    for i in range(n_agents):
        gpu_id = i % gpus_per_node
        device = f"{base_device}:{gpu_id}"
        llm = EmbeddedLLM(
            model=model,
            device=device,
            backend=backend,
            **kwargs,
        )
        llms.append(llm)
    return llms


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

    elif provider == "embedded":
        model = model or os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        device = os.getenv("LLM_DEVICE", "cuda:0")
        backend = os.getenv("LLM_BACKEND", "vllm")
        return EmbeddedLLM(model=model, device=device, backend=backend)

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            "Use 'mock', 'anthropic', 'openai_compatible', or 'embedded'"
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
