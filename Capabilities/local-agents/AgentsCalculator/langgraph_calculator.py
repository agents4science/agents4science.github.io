"""
Minimal agent example: an LLM that can use a calculator.

Supports three modes:
1. OPENAI_API_KEY set → uses OpenAI
2. FIRST_API_KEY set → uses FIRST (HPC inference service)
3. Neither set → uses mock responses to demonstrate the pattern
"""

import os
import re

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for any arithmetic."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_llm():
    """
    Get the appropriate LLM based on available credentials.

    Returns:
        tuple: (llm_instance or None, mode_description)
    """
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini"), "OpenAI (gpt-4o-mini)"

    if os.environ.get("FIRST_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.environ.get("FIRST_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
            api_key=os.environ["FIRST_API_KEY"],
            base_url=os.environ.get("FIRST_API_BASE", "https://api.first.example.com/v1"),
        ), f"FIRST ({os.environ.get('FIRST_MODEL', 'Llama-3.1-70B')})"

    return None, "Mock (no API key)"


def run_with_llm(llm, queries: list[str], mode: str):
    """Run queries using the LangGraph agent with a real LLM."""
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(llm, [calculate])

    print(f"Mode: {mode}")
    print("=" * 60)

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        for step in agent.stream({"messages": [HumanMessage(content=query)]}):
            if "agent" in step:
                msg = step["agent"]["messages"][0]
                if msg.content:
                    print(f"Agent: {msg.content}")
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"Agent calls: {tc['name']}({tc['args']})")
            elif "tools" in step:
                print(f"Tool result: {step['tools']['messages'][0].content}")


def run_mock(queries: list[str]):
    """Demonstrate the pattern with mock responses (no LLM API key required)."""
    print("Mode: Mock (no API key set)")
    print("=" * 60)
    print("Demonstrating the agent pattern with hardcoded responses.\n")
    print("Set OPENAI_API_KEY or FIRST_API_KEY to use a real LLM.\n")

    # Mock responses that demonstrate what the agent would do
    mock_flows = {
        "What is 347 * 892?": [
            ("Agent calls", "calculate", "347 * 892"),
            ("Tool result", "309524"),
            ("Agent", "347 × 892 = 309,524"),
        ],
        "If I have 1500 and spend 847, how much is left?": [
            ("Agent calls", "calculate", "1500 - 847"),
            ("Tool result", "653"),
            ("Agent", "If you have 1500 and spend 847, you have 653 left."),
        ],
    }

    for query in queries:
        print(f"Query: {query}")
        print("-" * 40)

        if query in mock_flows:
            for step in mock_flows[query]:
                if step[0] == "Agent calls":
                    print(f"Agent calls: {step[1]}({step[2]})")
                    # Actually call the tool to show it works
                    result = calculate.invoke(step[2])
                    print(f"Tool result: {result}")
                elif step[0] == "Agent":
                    print(f"Agent: {step[1]}")
        else:
            # For unknown queries, try to extract math and compute
            numbers = re.findall(r"\d+", query)
            if len(numbers) >= 2:
                expr = f"{numbers[0]} + {numbers[1]}"
                print(f"Agent calls: calculate({expr})")
                result = calculate.invoke(expr)
                print(f"Tool result: {result}")
                print(f"Agent: The result is {result}.")
            else:
                print("Agent: I need a mathematical expression to calculate.")
        print()


def main():
    queries = [
        "What is 347 * 892?",
        "If I have 1500 and spend 847, how much is left?",
    ]

    llm, mode = get_llm()

    if llm:
        run_with_llm(llm, queries, mode)
    else:
        run_mock(queries)


if __name__ == "__main__":
    main()
