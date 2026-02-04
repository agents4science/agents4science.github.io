"""
LLM configuration for the LangGraph scientific pipeline.

Supports four modes:
1. OPENAI_API_KEY set → uses OpenAI
2. FIRST_API_KEY set → uses FIRST (HPC inference service)
3. OLLAMA_MODEL set → uses Ollama (local LLM)
4. None of the above → uses mock responses
"""

import os
from typing import Any


# Global state for LLM mode
_llm_mode: str | None = None
_llm_reason: str | None = None


def get_llm_mode() -> str:
    """Get the current LLM mode."""
    global _llm_mode
    if _llm_mode is None:
        _detect_mode()
    return _llm_mode


def get_llm_reason() -> str:
    """Get the reason for the current LLM mode selection."""
    global _llm_reason
    if _llm_reason is None:
        _detect_mode()
    return _llm_reason


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """
    Get the appropriate LLM based on available credentials.

    Returns:
        LLM instance or None if in mock mode
    """
    global _llm_mode

    if _llm_mode is None:
        _detect_mode()

    if _llm_mode == "mock":
        return None

    # Lazy import to avoid dependency issues in mock mode
    from langchain_openai import ChatOpenAI

    if _llm_mode == "openai":
        return ChatOpenAI(model=model, temperature=temperature)

    if _llm_mode == "first":
        return ChatOpenAI(
            model=os.environ.get("FIRST_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
            api_key=os.environ["FIRST_API_KEY"],
            base_url=os.environ.get("FIRST_API_BASE", "https://api.first.example.com/v1"),
            temperature=temperature,
        )

    if _llm_mode == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return ChatOpenAI(
            model=os.environ["OLLAMA_MODEL"],
            api_key="ollama",
            base_url=f"{host}/v1",
            temperature=temperature,
        )

    return None


def _detect_mode():
    """Detect which LLM mode to use based on environment variables."""
    global _llm_mode, _llm_reason

    if os.environ.get("OPENAI_API_KEY"):
        _llm_mode = "openai"
        _llm_reason = "OPENAI_API_KEY found in environment"
    elif os.environ.get("FIRST_API_KEY"):
        _llm_mode = "first"
        model = os.environ.get("FIRST_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
        _llm_reason = f"FIRST_API_KEY found in environment (model: {model})"
    elif os.environ.get("OLLAMA_MODEL"):
        _llm_mode = "ollama"
        model = os.environ["OLLAMA_MODEL"]
        _llm_reason = f"OLLAMA_MODEL found in environment (model: {model})"
    else:
        _llm_mode = "mock"
        _llm_reason = "No API key or OLLAMA_MODEL found; using hardcoded responses"


def get_mode_description() -> str:
    """Get a human-readable description of the current mode."""
    mode = get_llm_mode()
    if mode == "openai":
        return "OpenAI (gpt-4o-mini)"
    elif mode == "first":
        model = os.environ.get("FIRST_MODEL", "Llama-3.1-70B")
        return f"FIRST ({model})"
    elif mode == "ollama":
        model = os.environ.get("OLLAMA_MODEL", "unknown")
        return f"Ollama ({model})"
    else:
        return "Mock"


def print_mode_info():
    """Print information about the selected LLM mode."""
    mode = get_mode_description()
    reason = get_llm_reason()
    print("=" * 60)
    print(f"LLM Mode: {mode}")
    print(f"  Reason: {reason}")
    print("=" * 60)
