# Minimal Agent Example: Calculator

The simplest possible agent: an LLM with a calculator tool.

**Code:** [github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsCalculator](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsCalculator)

## What It Does

1. User asks a math question
2. LLM recognizes it needs to calculate
3. LLM calls the `calculate` tool
4. Tool returns the result
5. LLM responds with the answer

## The Code

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression, {"__builtins__": {}}, {}))


llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, [calculate])

for step in agent.stream({"messages": [HumanMessage(content="What is 347 * 892?")]}):
    print(step)
```

## Running the Example

```bash
cd Capabilities/local-agents/AgentsCalculator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python langgraph_calculator.py
```

## LLM Configuration

The example supports three modes:

| Mode | Environment Variable | Description |
|------|---------------------|-------------|
| **OpenAI** | `OPENAI_API_KEY` | Uses OpenAI's gpt-4o-mini |
| **FIRST** | `FIRST_API_KEY` | Uses FIRST HPC inference service |
| **Mock** | (none) | Demonstrates pattern with hardcoded responses |

```bash
# OpenAI mode
export OPENAI_API_KEY=<your_key>
python langgraph_calculator.py

# FIRST mode (for HPC environments)
export FIRST_API_KEY=<your_token>
export FIRST_API_BASE=https://your-first-endpoint/v1
export FIRST_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
python langgraph_calculator.py

# Mock mode (no API key needed)
python langgraph_calculator.py
```

See [Argonne FIRST configuration](/Capabilities/local-agents/argonne-first/) for details on using Argonne's HPC inference service.

## Key Points

- **Tool definition**: The `@tool` decorator turns a function into an LLM-callable tool
- **Agent creation**: `create_react_agent` wires up the LLM with tools in a ReAct loop
- **Execution**: `agent.stream()` runs the agent and yields intermediate steps

For more complex workflows with branching, cycles, or custom state, use LangGraph's `StateGraph` directly (see [AgentsLangGraph](/Capabilities/local-agents/AgentsLangGraph/)).

## Requirements

- Python 3.10+
- LangGraph 1.0+
- OpenAI API key, FIRST token, or run in mock mode
