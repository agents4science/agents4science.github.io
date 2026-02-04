"""
Base agent class for agents4science.

Supports multiple LLM modes:
1. OPENAI_API_KEY set → uses OpenAI
2. FIRST_API_KEY set → uses FIRST (HPC inference service)
3. A4S_USE_INFERENCE=1 → uses Argonne Inference Service
4. None of the above → uses mock responses
"""

import os
import asyncio
import functools
from .logging_utils import get_logger
from openai import OpenAI, APIConnectionError, APITimeoutError

TIMEOUT = float(os.getenv("A4S_TIMEOUT", "60"))  # seconds

# Detect LLM mode
_llm_mode = None
_llm_reason = None
_client = None


def _detect_mode():
    """Detect which LLM mode to use based on environment variables."""
    global _llm_mode, _llm_reason, _client

    if os.environ.get("OPENAI_API_KEY"):
        _llm_mode = "openai"
        _llm_reason = "OPENAI_API_KEY found in environment"
        _client = OpenAI()

    elif os.environ.get("FIRST_API_KEY"):
        _llm_mode = "first"
        model = os.environ.get("FIRST_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
        base_url = os.environ.get("FIRST_API_BASE", "https://api.first.example.com/v1")
        _llm_reason = f"FIRST_API_KEY found in environment (model: {model})"
        _client = OpenAI(api_key=os.environ["FIRST_API_KEY"], base_url=base_url)

    elif os.getenv("A4S_USE_INFERENCE", "0") == "1":
        _llm_mode = "argonne"
        from inference_auth_token import get_access_token
        access_token = get_access_token()
        base_url = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
        _llm_reason = f"A4S_USE_INFERENCE=1 (Argonne Inference Service)"
        _client = OpenAI(api_key=access_token, base_url=base_url)

    else:
        _llm_mode = "mock"
        _llm_reason = "No OPENAI_API_KEY, FIRST_API_KEY, or A4S_USE_INFERENCE found; using mock responses"
        _client = None


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


def get_client():
    """Get the OpenAI client (or None for mock mode)."""
    global _client
    if _llm_mode is None:
        _detect_mode()
    return _client


def get_mode_description() -> str:
    """Get a human-readable description of the current mode."""
    mode = get_llm_mode()
    if mode == "openai":
        model = os.environ.get("A4S_MODEL", "gpt-4o-mini")
        return f"OpenAI ({model})"
    elif mode == "first":
        model = os.environ.get("FIRST_MODEL", "Llama-3.1-70B")
        return f"FIRST ({model})"
    elif mode == "argonne":
        model = os.environ.get("A4S_MODEL", "openai/gpt-oss-20b")
        return f"Argonne ({model})"
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


def _delay(env, default):
    try:
        return max(0.0, float(os.getenv(env, default)))
    except:
        return default


class Agent:
    def __init__(self, model=None, tools=None, memory=None, name=None):
        # Default model depends on mode
        if model is None:
            mode = get_llm_mode()
            if mode == "openai":
                model = os.environ.get("A4S_MODEL", "gpt-4o-mini")
            elif mode == "first":
                model = os.environ.get("FIRST_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
            else:
                model = os.environ.get("A4S_MODEL", "openai/gpt-oss-20b")

        self.model = model
        self.tools = tools or []
        self.memory = memory
        self.name = name or self.__class__.__name__
        self.log = get_logger(self.name)
        self.last_prompt = None
        self.last_response = None

    async def ask(self, prompt: str):
        self.last_prompt = prompt
        client = get_client()

        if client is not None:
            loop = asyncio.get_event_loop()

            def _infer():
                result = self._query_inference(prompt, client)
                self.last_response = result.get("response", "")
                return result

            result = await loop.run_in_executor(None, _infer)
        else:
            # Mock mode
            await asyncio.sleep(_delay("A4S_LATENCY", 0.4))
            result = {"response": f"[Mock: {self.model}] → {prompt[:160]}..."}
            self.last_response = result["response"]
        return result

    def _query_inference(self, prompt: str, client):
        """Blocking call executed in thread pool to query the inference service."""
        self.log.info(f"[bold cyan]{self.name}[/]: sending request to {self.model}")
        try:
            response = client.with_options(timeout=TIMEOUT).chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except APITimeoutError:
            self.log.error(f"Timeout after {TIMEOUT} seconds")
            return {"response": f"Timeout after {TIMEOUT}s"}
        except APIConnectionError as e:
            self.log.error("Connection error while contacting inference service")
            return {"response": "Connection error"}
        except Exception as e:
            self.log.error(f"Unexpected inference error: {e}")
            return {"response": f"Error: {e}"}

        if not hasattr(response, "choices") or response.choices is None:
            err_msg = getattr(response, "error", {}).get("message", "Unknown error")
            self.log.error(f"LLM returned error: {err_msg}")
            return {"response": f"Error: {err_msg}"}
        else:
            return {"response": response.choices[0].message.content}


def Tool(func):
    logger = get_logger(func.__name__)

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger.info(f"[cyan]{func.__name__}[/] start")
        await asyncio.sleep(_delay("A4S_TOOL_LATENCY", 0.2))
        out = await func(*args, **kwargs)
        logger.info(f"[cyan]{func.__name__}[/] done")
        return out

    return wrapper
