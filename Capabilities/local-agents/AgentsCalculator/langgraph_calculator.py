"""Minimal agent example: an LLM that can use a calculator (LangGraph version)."""
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for any arithmetic."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def main():
    llm = ChatOpenAI(model="gpt-4o-mini")
    agent = create_react_agent(llm, [calculate])

    # Example queries
    queries = [
        "What is 347 * 892?",
        "If I have 1500 and spend 847, how much is left?",
    ]

    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print("=" * 50)

        for step in agent.stream({"messages": [HumanMessage(content=query)]}):
            if "agent" in step:
                print(f"Agent: {step['agent']['messages'][0].content}")
            elif "tools" in step:
                print(f"Tool: {step['tools']['messages'][0].content}")


if __name__ == "__main__":
    main()
