# agents4science/base_agent_langchain.py
from __future__ import annotations
import importlib
from typing import Optional, List

from langchain_openai import ChatOpenAI

# ---- Prompts: LC 0.3+ -> langchain_core.prompts; LC 0.2 -> langchain.prompts
try:
    from langchain_core.prompts import ChatPromptTemplate
except ModuleNotFoundError:
    from langchain.prompts import ChatPromptTemplate  # LC 0.2.x

# ---- Tool typing (optional, for hints only)
try:
    from langchain_core.tools import BaseTool  # LC 0.3+
except ModuleNotFoundError:
    try:
        from langchain.tools.base import BaseTool  # LC 0.2.x
    except Exception:
        BaseTool = object  # fallback for typing only


def _resolve_agent_ctor():
    """
    Return (constructor, needs_executor=True) for a tool-calling agent across LC versions,
    or (None, False) if not available (fallback to plain LLM).
    Priority:
      1) langchain.agents.create_tool_calling_agent          (LC >= 0.3.6)
      2) langchain.agents.tool_calling.create_tool_calling_agent  (LC 0.3.0–0.3.5)
      3) langchain.agents.openai_functions.create_openai_functions_agent (LC 0.2.x)
    """
    # LC >= 0.3.6
    try:
        from langchain.agents import create_tool_calling_agent  # type: ignore
        return create_tool_calling_agent, True
    except Exception:
        pass

    # LC 0.3.0–0.3.5
    try:
        from langchain.agents.tool_calling import create_tool_calling_agent  # type: ignore
        return create_tool_calling_agent, True
    except Exception:
        pass

    # LC 0.2.x
    try:
        from langchain.agents.openai_functions import create_openai_functions_agent  # type: ignore
        return create_openai_functions_agent, True
    except Exception:
        pass

    return None, False


def _resolve_agent_executor():
    """
    Return AgentExecutor class if present, otherwise None (we'll fallback to plain LLM).
    """
    for mod_name in ("langchain.agents.executor", "langchain.agents"):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "AgentExecutor"):
                return getattr(mod, "AgentExecutor")
        except Exception:
            continue
    return None


CreateAgentCtor, _NEEDS_EXEC = _resolve_agent_ctor()
AgentExecutor = _resolve_agent_executor()


class LangAgent:
    """Version-robust LangChain agent wrapper with graceful fallback to plain LLM."""

    def __init__(
        self,
        name: str,
        role_description: str,
        tools: Optional[List[BaseTool]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.name = name
        self.role_description = role_description
        self.tools = tools or []
        self.model = model or "gpt-4o-mini"

        # LLM via langchain-openai
        self.llm = ChatOpenAI(model=self.model, temperature=temperature)

        # Shared prompt (system + user)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"You are the {self.name} agent. {self.role_description}"),
                ("human", "{input}"),
            ]
        )

        # Try to build a tool-calling agent + executor; otherwise fallback to simple LLM chain
        self._executor = None
        if CreateAgentCtor and AgentExecutor:
            try:
                agent = CreateAgentCtor(self.llm, self.tools, self.prompt)
                self._executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
            except Exception as e:
                # If construction fails for any reason, we’ll use plain LLM mode
                print(f"[{self.name}] Falling back to simple LLM mode: {e}")

    def act(self, input_text: str) -> str:
        """Run the agent; if no agent/executor, run the LLM with the prompt directly."""
        # Tool-calling agent path
        if self._executor is not None:
            try:
                result = self._executor.invoke({"input": input_text})
                if isinstance(result, dict) and "output" in result:
                    return result["output"]
                return str(result)
            except Exception as e:
                return f"[ERROR in {self.name}]: {e}"

        # Fallback: plain LLM with the same prompt
        try:
            chain = self.prompt | self.llm  # works in LC 0.2 and 0.3+
            res = chain.invoke({"input": input_text})
            # res is typically an AIMessage; extract .content if present
            content = getattr(res, "content", None)
            return content if content else str(res)
        except Exception as e:
            return f"[ERROR in {self.name} fallback]: {e}"

