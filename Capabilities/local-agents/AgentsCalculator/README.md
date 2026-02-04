# Minimal Agent Example: Calculator

The simplest possible agent: an LLM with a calculator tool. Implementations in both LangChain and LangGraph.

**Code:** [github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsCalculator](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsCalculator)

## What It Does

1. User asks a math question
2. LLM recognizes it needs to calculate
3. LLM calls the `calculate` tool
4. Tool returns the result
5. LLM responds with the answer

## LangChain Version

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression, {"__builtins__": {}}, {}))


llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the calculate tool for math."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, [calculate], prompt)
executor = AgentExecutor(agent=agent, tools=[calculate])

result = executor.invoke({"input": "What is 347 * 892?"})
print(result["output"])  # 309524
```

## LangGraph Version

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

## Running the Examples

```bash
cd Capabilities/local-agents/AgentsCalculator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=<your_key>

# LangChain version
python langchain_calculator.py

# LangGraph version
python langgraph_calculator.py
```

## Comparison

| Aspect | LangChain | LangGraph |
|--------|-----------|-----------|
| Agent creation | `create_tool_calling_agent` + `AgentExecutor` | `create_react_agent` |
| Prompt | Explicit `ChatPromptTemplate` | Built-in (customizable via `state_modifier`) |
| Execution | `executor.invoke()` | `agent.stream()` or `agent.invoke()` |
| Output | `result["output"]` | Message stream |
| Lines of code | ~15 | ~10 |

LangGraph's `create_react_agent` is a concise prebuilt for simple cases. For more control over prompts or flow, both frameworks offer lower-level APIs—LangGraph's `StateGraph` being particularly powerful for complex workflows.

## Requirements

- Python 3.10+
- LangChain 1.0+, LangGraph 1.0+
- OpenAI API key
