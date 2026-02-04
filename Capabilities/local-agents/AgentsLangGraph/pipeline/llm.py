"""
LLM configuration for the LangGraph scientific pipeline.

Supports three modes:
1. OPENAI_API_KEY set → uses OpenAI
2. FIRST_API_KEY set → uses FIRST (HPC inference service)
3. Neither set → uses mock responses
"""

import os
from typing import Any


# Global state for LLM mode
_llm_mode: str | None = None
_llm_instance: Any = None


def get_llm_mode() -> str:
    """Get the current LLM mode."""
    global _llm_mode
    if _llm_mode is None:
        _detect_mode()
    return _llm_mode


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """
    Get the appropriate LLM based on available credentials.

    Returns:
        LLM instance or None if in mock mode
    """
    global _llm_mode, _llm_instance

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

    return None


def _detect_mode():
    """Detect which LLM mode to use based on environment variables."""
    global _llm_mode

    if os.environ.get("OPENAI_API_KEY"):
        _llm_mode = "openai"
    elif os.environ.get("FIRST_API_KEY"):
        _llm_mode = "first"
    else:
        _llm_mode = "mock"


def get_mode_description() -> str:
    """Get a human-readable description of the current mode."""
    mode = get_llm_mode()
    if mode == "openai":
        return "OpenAI (gpt-4o-mini)"
    elif mode == "first":
        model = os.environ.get("FIRST_MODEL", "Llama-3.1-70B")
        return f"FIRST ({model})"
    else:
        return "Mock (no API key set)"
