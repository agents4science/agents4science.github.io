"""
Database Agent: Natural language queries on scientific datasets.

Supports four modes:
1. OPENAI_API_KEY set -> uses OpenAI
2. FIRST_API_KEY set -> uses FIRST (HPC inference service)
3. OLLAMA_MODEL set -> uses Ollama (local LLM)
4. None of the above -> uses mock responses to demonstrate the pattern
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# Data directory containing sample dataset
DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "catalyst_experiments.csv"

# Load the dataset
print(f"Loading dataset from {DATA_FILE.name}...")
df = pd.read_csv(DATA_FILE)


def get_llm():
    """Get the appropriate LLM based on available credentials."""
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

    return (
        None,
        "Mock",
        "No API key or OLLAMA_MODEL found; using hardcoded responses",
    )


def print_mode_info(mode: str, reason: str):
    """Print information about the selected LLM mode."""
    print("=" * 60)
    print(f"LLM Mode: {mode}")
    print(f"  Reason: {reason}")
    print("=" * 60)


@tool
def describe_columns() -> str:
    """Get information about available columns in the catalyst experiment dataset."""
    info = []
    info.append("Columns in the dataset:")
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "object":
            unique = df[col].unique()[:5]
            info.append(f"  - {col} (text): e.g., {', '.join(map(str, unique))}")
        else:
            info.append(f"  - {col} ({dtype}): range {df[col].min()} to {df[col].max()}")
    info.append(f"\nTotal rows: {len(df)}")
    return "\n".join(info)


@tool
def query_data(pandas_query: str) -> str:
    """
    Execute a pandas query on the catalyst dataset.
    Use df.query() syntax, e.g.: "temperature_c == 25" or "efficiency_pct > 80"
    """
    try:
        result = df.query(pandas_query)
        if len(result) == 0:
            return "No rows match the query."
        return result.to_string(index=False)
    except Exception as e:
        return f"Query error: {e}"


@tool
def get_statistics(column: str) -> str:
    """Get summary statistics for a numeric column (e.g., efficiency_pct, temperature_c)."""
    if column not in df.columns:
        return f"Column '{column}' not found. Available: {', '.join(df.columns)}"
    if df[column].dtype == "object":
        return f"Value counts for {column}:\n{df[column].value_counts().to_string()}"
    return f"Statistics for {column}:\n{df[column].describe().to_string()}"


@tool
def get_top_n(column: str, n: int = 5, ascending: bool = False) -> str:
    """Get the top (or bottom) N rows sorted by a column."""
    try:
        result = df.nlargest(n, column) if not ascending else df.nsmallest(n, column)
        return result.to_string(index=False)
    except Exception as e:
        return f"Error: {e}"


def run_with_llm(llm, question: str):
    """Run the database agent with a real LLM."""
    from langgraph.prebuilt import create_react_agent

    print(f"\nQuestion: {question}")
    print("-" * 60)

    print("Dataset preview:")
    print(df.head(3).to_string(index=False))
    print(f"... ({len(df)} total rows)\n")

    agent = create_react_agent(llm, [describe_columns, query_data, get_statistics, get_top_n])

    for step in agent.stream({"messages": [HumanMessage(content=question)]}):
        if "agent" in step:
            msg = step["agent"]["messages"][0]
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc["name"]
                    args = tc["args"]
                    print(f"Agent calls: {tool_name}({args})")
            if msg.content:
                print(f"\nAnswer: {msg.content}")
        elif "tools" in step:
            content = step["tools"]["messages"][0].content
            preview = content[:300] + "..." if len(content) > 300 else content
            print(f"Result:\n{preview}\n")


def run_mock(question: str):
    """Demonstrate the database agent pattern with mock responses."""
    print("\nDemonstrating database query pattern with mock responses.")
    print("Set OPENAI_API_KEY to use a real LLM.\n")

    print(f"Question: {question}")
    print("-" * 60)

    print("Dataset preview:")
    print(df.head(3).to_string(index=False))
    print(f"... ({len(df)} total rows)\n")

    question_lower = question.lower()

    if "highest efficiency" in question_lower or "best" in question_lower:
        print("Agent calls: get_top_n({'column': 'efficiency_pct', 'n': 5})")
        result = get_top_n.invoke({"column": "efficiency_pct", "n": 5})
        print(f"Result:\n{result}\n")
        answer = """Based on the data, the catalysts with highest efficiency are:

1. **Au-nanoparticles**: 97.2% efficiency (produces CO at 25°C)
2. **Ag-nanoparticles**: 94.5% efficiency (produces CO at 25°C)
3. **Fe-SAC at 50°C**: 92.4% efficiency (produces CO)

Note that while gold and silver achieve the highest efficiencies, they primarily
produce CO. For multi-carbon products like ethylene (C2H4), copper-based
catalysts are the best option with ~62-68% efficiency."""

    elif "room temperature" in question_lower or "25" in question_lower:
        print("Agent calls: query_data({'pandas_query': 'temperature_c == 25'})")
        result = query_data.invoke({"pandas_query": "temperature_c == 25"})
        print(f"Result:\n{result}\n")
        answer = """At room temperature (25°C), the dataset shows several catalyst options:

- **Highest efficiency**: Au-nanoparticles (97.2%) and Ag-nanoparticles (94.5%)
  but these only produce CO
- **For hydrocarbons**: Cu-nanoparticles achieve 62.3% for ethylene (C2H4)
- **Most stable**: TiO2-photocatalyst runs for 1000 hours, but only 0.8% efficiency

The trade-off is between efficiency and product value—noble metals are efficient
but produce low-value CO, while copper produces valuable hydrocarbons at lower efficiency."""

    elif "stable" in question_lower or "stability" in question_lower:
        print("Agent calls: get_top_n({'column': 'stability_hrs', 'n': 5})")
        result = get_top_n.invoke({"column": "stability_hrs", "n": 5})
        print(f"Result:\n{result}\n")
        answer = """The most stable catalysts in the dataset are:

1. **TiO2-photocatalyst**: 1000 hours (but very low efficiency at 0.8%)
2. **Ag-nanoparticles**: 500 hours with 94.5% efficiency
3. **Au-nanoparticles**: 480 hours with 97.2% efficiency

For practical applications, silver nanoparticles offer the best balance of
high efficiency and long stability for CO production."""

    else:
        print("Agent calls: describe_columns()")
        result = describe_columns.invoke({})
        print(f"Result:\n{result}\n")
        answer = f"""The catalyst experiment dataset contains {len(df)} experiments with the
following information:
- **catalyst**: The catalyst material tested
- **temperature_c**: Reaction temperature (25-250°C)
- **efficiency_pct**: Conversion efficiency (0.8-97.2%)
- **product**: Main product (CO, CH4, C2H4, CH3OH)
- **stability_hrs**: Hours of stable operation

You can ask about specific catalysts, temperatures, or products."""

    print(f"Answer: {answer}")


def main():
    parser = argparse.ArgumentParser(description="Database Agent for scientific data")
    parser.add_argument(
        "--question",
        default="Which catalyst has the highest efficiency at room temperature?",
        help="Question to ask about the data",
    )
    args = parser.parse_args()

    llm, mode, reason = get_llm()
    print_mode_info(mode, reason)

    if llm:
        run_with_llm(llm, args.question)
    else:
        run_mock(args.question)


if __name__ == "__main__":
    main()
