"""
Minimal agent example: an LLM that can use a calculator.

Supports four modes:
1. OPENAI_API_KEY set → uses OpenAI
2. FIRST_API_KEY set → uses FIRST (HPC inference service)
3. OLLAMA_MODEL set → uses Ollama (local LLM)
4. None of the above → uses mock responses to demonstrate the pattern
"""

import os
import re

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for any arithmetic."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

class ToolEnabledFakeChatModel(GenericFakeChatModel):
    """A fake model that implements bind_tools."""
    def bind_tools(
        self, 
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]], 
        **kwargs: Any
    ) -> Any:
        return self

def get_llm():
    """
    Get the appropriate LLM based on available credentials.

    Returns:
        tuple: (llm_instance or None, mode_name, reason)
    """

    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return (
            ChatOpenAI(model="gpt-4o-mini"),
            "OpenAI",
            "OPENAI_API_KEY found in environment",
        )

    if os.environ.get("FIRST_API_KEY"):
        from langchain_openai import ChatOpenAI
        model = os.environ.get("FIRST_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
        base_url = os.environ.get("FIRST_API_BASE", "https://api.first.example.com/v1")
        return (
            ChatOpenAI(model=model, api_key=os.environ["FIRST_API_KEY"], base_url=base_url),
            "FIRST",
            f"FIRST_API_KEY found in environment (model: {model})",
        )

    if os.environ.get("OLLAMA_MODEL"):
        from langchain_openai import ChatOpenAI
        model = os.environ["OLLAMA_MODEL"]
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        base_url = f"{host}/v1"
        return (
            ChatOpenAI(model=model, api_key="ollama", base_url=base_url),
            "Ollama",
            f"OLLAMA_MODEL found in environment (model: {model})",
        )

    # No API key or local model; using hardcoded responses
    responses = [
        AIMessage(content="", tool_calls=[{"name": "calculate", "args": {"expression": "347 * 892"}, "id": "call_1"}]),
        AIMessage(content="347 * 892 is 309,524."),
        AIMessage(content="", tool_calls=[{"name": "calculate", "args": {"expression": "1500 - 847"}, "id": "call_2"}]),
        AIMessage(content="You have 653 left.")

    ]
    fake_llm = ToolEnabledFakeChatModel(messages=iter(responses))

    return (
        fake_llm,
        "FakeModel",
        "No API key or OLLAMA_MODEL found; using hardcoded responses"
    )


def print_mode_info(mode: str, reason: str):
    """Print information about the selected LLM mode."""
    print("=" * 60)
    print(f"LLM Mode: {mode}")
    print(f"  Reason: {reason}")
    print("=" * 60)


def run_with_llm(llm, queries: list[str]):
    """Run queries using the LangGraph agent with a real LLM."""
    from langchain.agents import create_agent
    
    agent = create_agent(llm, [calculate])

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        for step in agent.stream({"messages": [HumanMessage(content=query)]}):
            if "agent" in step:
                msg = step["agent"]["messages"][0]
                if msg.content:
                    print(f"Agent: {msg.content}")
                for tc in getattr(msg, "tool_calls", []):
                    print(f"Agent calls: {tc['name']}({tc['args']})")
            elif "tools" in step:
                print(f"Tool result: {step['tools']['messages'][0].content}")


def main():
    queries = [
        "What is 347 * 892?",
        "If I have 1500 and spend 847, how much is left?",
    ]

    llm, mode, reason = get_llm()
    print_mode_info(mode, reason)

    run_with_llm(llm, queries)


if __name__ == "__main__":
    main()
