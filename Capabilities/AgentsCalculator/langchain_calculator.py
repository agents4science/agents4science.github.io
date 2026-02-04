"""Minimal agent example: an LLM that can use a calculator."""
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate


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

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the calculate tool for math."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, [calculate], prompt)
    executor = AgentExecutor(agent=agent, tools=[calculate], verbose=True)

    # Example queries
    queries = [
        "What is 347 * 892?",
        "If I have 1500 and spend 847, how much is left?",
    ]

    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print("=" * 50)
        result = executor.invoke({"input": query})
        print(f"Answer: {result['output']}")


if __name__ == "__main__":
    main()
