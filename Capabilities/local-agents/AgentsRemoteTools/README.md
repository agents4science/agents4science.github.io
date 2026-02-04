# Academy Remote Tools

Demonstrates how Academy agents can provide tools that other agents call remotely.

**Code:** [View on GitHub](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsRemoteTools)

## The Pattern

This example shows a key pattern for distributed scientific computing: **tools as agent actions**.

```
+---------------+                    +------------------+
|  Coordinator  |                    |  ToolProvider    |
|               |                    |                  |
|  execute_     |  -- simulation --> |  run_simulation()|
|  workflow()   |                    |        |         |
|       |       |                    |        v         |
|       |       |  <-- results ----  |   compute...     |
|       |       |                    |                  |
|       |       |  -- analysis ----> |  analyze_data()  |
|       v       |                    |        |         |
|   aggregate   |  <-- results ----  |        v         |
|    results    |                    |    analyze...    |
+---------------+                    +------------------+
        ^                                     |
        |                                     |
        +---- Academy Message Exchange -------+
```

**Why this matters:**
- **Separation of concerns**: Coordinator handles workflow logic; ToolProvider handles compute
- **Scalability**: ToolProvider can run on HPC, cloud, or lab instruments
- **Security**: Academy's identity and policy layer controls who can call what
- **Flexibility**: Same coordinator can work with different tool providers

## Running the Example

```bash
cd Capabilities/local-agents/AgentsRemoteTools
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Custom task:

```bash
python main.py --task "analyze protein folding data"
```

## Example Output

```
============================================================
ACADEMY REMOTE TOOLS EXAMPLE
============================================================
Task: optimize molecular structure for catalysis
------------------------------------------------------------
16:30:00 [academy.remote] Launched ToolProvider: ...
16:30:00 [academy.remote] Launched Coordinator: ...
16:30:00 [academy.remote] [Coordinator] Tool provider configured
16:30:00 [academy.remote] [Coordinator] Starting workflow for: optimize...
16:30:00 [academy.remote] [Coordinator] Tool provider status: ready
16:30:00 [academy.remote] [Coordinator] Step 1: Running initial simulation
16:30:00 [academy.remote] [ToolProvider] Starting simulation SIM-0001
16:30:00 [academy.remote] [ToolProvider] Completed SIM-0001: energy=-125.32
16:30:00 [academy.remote] [Coordinator] Step 2: Analyzing data
16:30:00 [academy.remote] [ToolProvider] Analyzing SIM-0001 with statistical
16:30:01 [academy.remote] [Coordinator] Step 3: Running optimized simulation
16:30:01 [academy.remote] [ToolProvider] Starting simulation SIM-0002
16:30:01 [academy.remote] [ToolProvider] Completed SIM-0002: energy=-129.47
16:30:01 [academy.remote] [Coordinator] Workflow complete: improvement=4.15
------------------------------------------------------------
WORKFLOW RESULTS
------------------------------------------------------------
Status: completed
Steps completed: 3

Summary:
  Initial energy:   -125.32
  Optimized energy: -129.47
  Improvement:      4.15
============================================================
```

## Key Concepts

### 1. Tools as Agent Actions

Any `@action` method can be called remotely by other agents:

```python
class ToolProviderAgent(Agent):
    @action
    async def run_simulation(self, parameters: dict) -> dict:
        # Expensive computation here
        return {"energy": -127.5, "status": "completed"}
```

### 2. Handle-Based Tool Access

The coordinator receives a Handle to the tool provider and calls its actions:

```python
class CoordinatorAgent(Agent):
    def __init__(self):
        self._tool_provider: Handle | None = None

    @action
    async def set_tool_provider(self, provider: Handle) -> None:
        self._tool_provider = provider

    @action
    async def execute_workflow(self, task: str) -> dict:
        # Call tools on the remote provider
        result = await self._tool_provider.run_simulation(
            parameters={"temperature": 300.0}
        )
        return result
```

### 3. Multi-Step Workflows

The coordinator can chain multiple tool calls:

```python
# Step 1: Run simulation
sim_result = await self._tool_provider.run_simulation(params)

# Step 2: Analyze results
analysis = await self._tool_provider.analyze_data(sim_result["job_id"])

# Step 3: Optimized run based on analysis
optimized = await self._tool_provider.run_simulation(new_params)
```

## Production Deployment

In production, this pattern enables powerful distributed scenarios:

| Component | Development | Production |
|-----------|-------------|------------|
| **ToolProvider** | Same machine | HPC login node, instrument computer, cloud VM |
| **Coordinator** | Same machine | User workstation, workflow service, AI agent |
| **Exchange** | LocalExchangeFactory | GlobusComputeExchange, RedisExchange |

The code remains identical - only the exchange factory changes:

```python
# Development
factory = LocalExchangeFactory()

# Production (example - actual API may vary)
factory = GlobusComputeExchange(endpoint_id="hpc-cluster-xyz")
```

## Comparison to Other Patterns

| Pattern | Coordination | Best For |
|---------|--------------|----------|
| **Remote Tools** (this) | Explicit tool calls | HPC jobs, instruments, expensive compute |
| **Pipeline** | Agent-to-agent forwarding | Sequential processing |
| **Hub-and-Spoke** | Central orchestration | Simple workflows |

## Related Examples

**Prerequisites:**
- [AgentsAcademyBasic](/Capabilities/local-agents/AgentsAcademyBasic/) - Academy fundamentals (start here if new)

**Next steps:**
- [AgentsHybrid](/Capabilities/local-agents/AgentsHybrid/) - Add LLM reasoning to Academy agents
- [AgentsPersistent](/Capabilities/local-agents/AgentsPersistent/) - Checkpoint and resume workflows
- [AgentsFederated](/Capabilities/local-agents/AgentsFederated/) - Cross-institutional collaboration
- [Federated Agent Execution](/Capabilities/federated-agents/) - Production deployment on HPC
- [Governed Tool Use](/Capabilities/governed-tool-use/) - Add policy enforcement

## Requirements

- Python 3.10+
- academy-py
