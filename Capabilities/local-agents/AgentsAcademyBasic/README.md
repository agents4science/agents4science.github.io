# Minimal Academy Example

The simplest possible Academy agent setup: two agents that communicate.

**Code:** [View on GitHub](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsAcademyBasic)

## What It Does

Demonstrates Academy basics with just two agents:

1. **RequesterAgent**: Sends a calculation request to the Calculator
2. **CalculatorAgent**: Performs the calculation, returns the result

This is the "Hello World" of Academy - understand this before moving to complex pipelines.

## Running the Example

```bash
cd Capabilities/local-agents/AgentsAcademyBasic
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Custom expression:

```bash
python main.py --expression "2 ** 10"
```

## Example Output

```
==================================================
MINIMAL ACADEMY EXAMPLE
==================================================
Expression: 42 * 17
--------------------------------------------------
16:57:51 [academy.basic] Launched Calculator: ...
16:57:51 [academy.basic] Launched Requester: ...
16:57:51 [academy.basic] Requester: Calculator handle received
16:57:51 [academy.basic] Requester sending: 42 * 17
16:57:51 [academy.basic] Calculator received: 42 * 17
16:57:51 [academy.basic] Calculator computed: 42 * 17 = 714
16:57:51 [academy.basic] Requester received: 42 * 17 = 714
--------------------------------------------------
RESULT: 42 * 17 = 714
==================================================
```

## Architecture

```
+-------------+                  +---------------+
|  Requester  |                  |  Calculator   |
|             |                  |               |
|  request_   |  --- call -----> |  calculate()  |
|  calculation|                  |       |       |
|      |      |                  |       v       |
|      v      |  <-- return ---  |    eval()     |
|   result    |                  |               |
+-------------+                  +---------------+
      ^                                  |
      |                                  |
      +---- Academy Message Exchange ----+
```

## Key Concepts

### 1. Agents are Classes

```python
from academy.agent import Agent, action

class CalculatorAgent(Agent):
    @action
    async def calculate(self, expression: str) -> str:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
```

### 2. Actions are Async Methods

The `@action` decorator marks methods that can be called remotely:

```python
@action
async def calculate(self, expression: str) -> str:
    # This can be called by other agents or the main process
    return f"Result: {eval(expression)}"
```

### 3. Manager Launches Agents

```python
async with await Manager.from_exchange_factory(
    factory=LocalExchangeFactory(),
) as manager:
    calculator = await manager.launch(CalculatorAgent)
```

### 4. Handles Enable Communication

`manager.launch()` returns a Handle (proxy) for the agent. Handles can be passed to other agents:

```python
# Main process passes calculator handle to requester
await requester.set_calculator(calculator)

# Inside RequesterAgent, store and use the handle
@action
async def set_calculator(self, calculator: Handle) -> None:
    self._calculator = calculator

@action
async def request_calculation(self, expression: str) -> str:
    # Call calculator via its handle
    result = await self._calculator.calculate(expression)
    return result
```

### 5. Local vs Remote Exchange

- `LocalExchangeFactory()` - agents run in same process (development)
- Remote exchanges - agents run across machines (production)

Same code works with both!

## Comparison: Academy vs LangGraph

| Aspect | Academy | LangGraph |
|--------|---------|-----------|
| **Agent model** | Independent processes/actors | Functions in a graph |
| **Communication** | Message passing via Handles | Shared state |
| **Scaling** | Distributed by design | Single process (typically) |
| **Best for** | Multi-system, HPC, federation | LLM reasoning chains |

## Next Steps

After understanding this example, explore:

- [AgentsAcademy](/Capabilities/local-agents/AgentsAcademy/) - 5-agent pipeline
- [AgentsAcademyHubSpoke](/Capabilities/local-agents/AgentsAcademyHubSpoke/) - Hub-and-spoke pattern
- [AgentsAcademyDashboard](/Capabilities/local-agents/AgentsAcademyDashboard/) - Rich TUI

## Requirements

- Python 3.10+
- academy-py
