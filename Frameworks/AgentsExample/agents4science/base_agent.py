import os, asyncio, functools, aiohttp
from .logging_utils import get_logger
from openai import OpenAI, APIConnectionError, APITimeoutError

USE_INFERENCE = os.getenv("A4S_USE_INFERENCE", "0") == "1"
TIMEOUT = float(os.getenv("A4S_TIMEOUT", "60"))   # seconds

if USE_INFERENCE:
    from inference_auth_token import get_access_token
    # Get your access token
    access_token = get_access_token()
    API_KEY=access_token
    BASE_URL="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if USE_INFERENCE else None

def _delay(env, default):
    try: return max(0.0, float(os.getenv(env, default)))
    except: return default

class Agent:
    def __init__(self, model='openai/gpt-oss-20b', tools=None, memory=None, name=None):
        self.model = model
        self.tools = tools or []
        self.memory = memory
        self.name = name or self.__class__.__name__
        self.log = get_logger(self.name)
        self.last_prompt = None
        self.last_response = None

    async def ask(self, prompt: str):
        self.last_prompt = prompt
        if USE_INFERENCE and client is not None:
            loop = asyncio.get_event_loop()

            def _infer():
                result = self._query_inference(prompt)
                self.last_response = result.get("response", "")
                return result

            result = await loop.run_in_executor(None, _infer)
        else:
            await asyncio.sleep(_delay("A4S_LATENCY", 0.4))
            result = {"response": f"[{self.model}] → {prompt[:160]}..."}
            self.last_response = result["response"]
        return result

    def _query_inference(self, prompt: str):
        """Blocking call executed in thread pool to query the inference service."""
        self.log.info(f"[bold cyan]{self.name}[/]: sending request to AIS model {self.model}")
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
            self.log.error(f"AIS returned error: {err_msg}")
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
