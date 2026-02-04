# Academy + LangGraph Hybrid

Combines Academy for distributed agent coordination with LangGraph for LLM-powered reasoning.

**Code:** [View on GitHub](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsHybrid)

## Why Hybrid?

Academy and LangGraph serve different purposes:

| Framework | Strength | Use For |
|-----------|----------|---------|
| **Academy** | Distributed execution, identity, messaging | Agent coordination across machines |
| **LangGraph** | LLM reasoning, tool calling, state graphs | Intelligent decision-making within agents |

Together they enable **distributed intelligent agents**: agents that can run anywhere (Academy) and think intelligently (LangGraph).

## Architecture

```
+--------------------------------------------------+
|                 Academy Runtime                   |
|                                                  |
|  +-----------------+      +-----------------+    |
|  | ResearcherAgent |      |  PlannerAgent   |    |
|  |                 |      |                 |    |
|  |  +----------+   | msg  |  +----------+   |    |
|  |  | LangGraph|   | ---> |  | LangGraph|   |    |
|  |  |  Agent   |   |      |  |  Agent   |   |    |
|  |  |  (LLM +  |   |      |  |  (LLM +  |   |    |
|  |  |  tools)  |   |      |  |  tools)  |   |    |
|  |  +----------+   |      |  +----------+   |    |
|  +-----------------+      +-----------------+    |
+--------------------------------------------------+
```

## Running the Example

```bash
cd Capabilities/local-agents/AgentsHybrid
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Mock mode (no API key needed)
python main.py

# With OpenAI
export OPENAI_API_KEY="your-key"
python main.py --llm openai

# With local Ollama
python main.py --llm ollama
```

## Example Output

```
============================================================
ACADEMY + LANGGRAPH HYBRID EXAMPLE
============================================================
Task: analyze catalysts for CO2 conversion at room temperature
LLM: mock
------------------------------------------------------------
17:00:00 [academy.hybrid] Launched Researcher: ...
17:00:00 [academy.hybrid] Launched Planner: ...
17:00:00 [academy.hybrid] [Researcher] Connected to Planner
17:00:00 [academy.hybrid] Starting hybrid workflow...
17:00:00 [academy.hybrid] [Researcher] Starting research: analyze catalysts...
17:00:00 [academy.hybrid] [Researcher] Research complete
17:00:00 [academy.hybrid] [Researcher] Forwarding findings to Planner
17:00:00 [academy.hybrid] [Planner] Creating plan from findings
17:00:00 [academy.hybrid] [Planner] Plan created
------------------------------------------------------------
WORKFLOW RESULTS
------------------------------------------------------------

### Research Findings ###
Based on my analysis:
1. **Key Finding**: The data shows promising catalyst candidates
2. **Recommendation**: Proceed with computational screening
3. **Next Steps**: Run DFT calculations on top 5 candidates

### Research Plan ###
Based on my analysis:
1. **Key Finding**: The data shows promising catalyst candidates
2. **Recommendation**: Proceed with computational screening
3. **Next Steps**: Run DFT calculations on top 5 candidates
============================================================
```

## Key Concepts

### 1. Academy Agent with LangGraph Inside

```python
class ResearcherAgent(Agent):
    """Academy agent using LangGraph for reasoning."""

    def __init__(self, llm_type: str = "mock"):
        super().__init__()
        self._langgraph_agent = None
        self.llm_type = llm_type

    def _init_langgraph(self):
        """Lazy initialization of LangGraph."""
        if self._langgraph_agent is None:
            llm = get_llm(self.llm_type)
            tools = [search_literature, analyze_results]
            self._langgraph_agent = create_react_agent(llm, tools)

    @action
    async def research(self, task: str) -> dict:
        self._init_langgraph()

        # Use LangGraph for LLM reasoning
        result = self._langgraph_agent.invoke({
            "messages": [HumanMessage(content=task)]
        })
        return {"findings": result["messages"][-1].content}
```

### 2. Agent-to-Agent Communication

Academy handles messaging between agents:

```python
@action
async def set_planner(self, planner: Handle) -> None:
    self._planner = planner

@action
async def research(self, task: str) -> dict:
    # ... LangGraph reasoning ...

    # Forward to next agent via Academy messaging
    if self._planner:
        await self._planner.plan(findings)
```

### 3. LangGraph Tools

Define tools that the LLM can use:

```python
@tool
def search_literature(query: str) -> str:
    """Search scientific literature."""
    # Real implementation would call APIs
    return f"Found papers about {query}"

@tool
def run_calculation(structure: str) -> str:
    """Run computational chemistry calculation."""
    # Real implementation would submit HPC job
    return f"Energy: -127.5 eV"
```

## When to Use This Pattern

| Scenario | Use Hybrid? |
|----------|-------------|
| Agents need LLM reasoning | Yes |
| Agents run on different machines | Yes |
| Simple message passing only | No, use Academy alone |
| Single-process LLM workflow | No, use LangGraph alone |

## Comparison to Alternatives

| Approach | Coordination | Intelligence | Distribution |
|----------|--------------|--------------|--------------|
| **LangGraph only** | Graph edges | LLM reasoning | Single process |
| **Academy only** | Message passing | Rule-based | Distributed |
| **Hybrid** (this) | Academy messages | LangGraph LLM | Distributed |

## Production Considerations

1. **LLM Placement**: Put LLM calls on GPU-enabled nodes
2. **Caching**: Cache LLM responses for reproducibility
3. **Fallbacks**: Handle LLM failures gracefully (mock fallback)
4. **Cost**: LLM calls have latency/cost; batch when possible

## Related Examples

**Prerequisites:**
- [AgentsAcademyBasic](/Capabilities/local-agents/AgentsAcademyBasic/) - Academy fundamentals
- [AgentsRemoteTools](/Capabilities/local-agents/AgentsRemoteTools/) - Remote tool pattern
- [AgentsCalculator](/Capabilities/local-agents/AgentsCalculator/) - LangGraph basics

**Next steps:**
- [AgentsPersistent](/Capabilities/local-agents/AgentsPersistent/) - Add checkpoint/resume to your hybrid agents
- [AgentsFederated](/Capabilities/local-agents/AgentsFederated/) - Cross-institutional collaboration
- [CharacterizeChemicals](/Capabilities/federated-agents/CharacterizeChemicals/) - Real-world Academy + LLM example

**Alternatives:**
- [AgentsLangGraph](/Capabilities/local-agents/AgentsLangGraph/) - Pure LangGraph approach (no Academy)

## Requirements

- Python 3.10+
- academy-py
- langchain, langgraph
- langchain-openai (for OpenAI) or langchain-ollama (for Ollama)
