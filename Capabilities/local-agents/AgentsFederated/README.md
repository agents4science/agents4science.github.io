# Academy Federated Collaboration

Demonstrates how Academy agents at different institutions can collaborate on scientific tasks.

**Code:** [View on GitHub](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsFederated)

## The Vision

Scientific research increasingly requires capabilities distributed across multiple institutions. This example shows how Academy enables:

- **Cross-institutional collaboration**: Agents at different labs work together
- **Specialized capabilities**: Each institution contributes unique resources
- **Coordinated workflows**: A coordinator orchestrates the overall task
- **Secure identity**: Academy handles authentication and authorization

## Architecture

```
                    +------------------+
                    |   Coordinator    |
                    | (orchestrates)   |
                    +--------+---------+
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
+------------------+ +------------------+ +------------------+
|       ANL        | |       ORNL       | |       LBNL       |
| (Aurora compute) | |  (SNS/HFIR data) | |   (ML analysis)  |
+------------------+ +------------------+ +------------------+
```

**In production:**
- Each institution runs its own agents behind institutional firewalls
- Academy handles authentication via federated identity (e.g., Globus Auth)
- Policies control what data/compute can be shared

## Running the Example

```bash
cd Capabilities/local-agents/AgentsFederated
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py

# Custom task
python main.py --task "study quantum materials for energy storage"
```

## Example Output

```
======================================================================
ACADEMY FEDERATED COLLABORATION EXAMPLE
======================================================================
Task: characterize novel superconducting materials
----------------------------------------------------------------------
Simulating agents at:
  - ANL (Argonne): Compute capabilities (Aurora)
  - ORNL (Oak Ridge): Data capabilities (SNS, HFIR)
  - LBNL (Berkeley): Analysis capabilities (Perlmutter)
----------------------------------------------------------------------
17:00:00 [academy.federated] Launched all institutional agents
17:00:00 [academy.federated] [Coordinator] Registered agent from ANL
17:00:00 [academy.federated] [Coordinator] Registered agent from ORNL
17:00:00 [academy.federated] [Coordinator] Registered agent from LBNL
17:00:00 [academy.federated] [Coordinator] Starting federated workflow...
17:00:00 [academy.federated] [Coordinator] Step 1: Querying institutional capabilities
17:00:00 [academy.federated] [Coordinator] ANL: ['DFT', 'MD', 'ML_inference']
17:00:00 [academy.federated] [Coordinator] ORNL: ['neutron_data', 'materials_db', 'data_curation']
17:00:00 [academy.federated] [Coordinator] LBNL: ['ML_analysis', 'beamline_data', 'visualization']
17:00:00 [academy.federated] [Coordinator] Step 2: Querying data from ORNL
17:00:00 [academy.federated] [ORNL] Found 1247 records
17:00:00 [academy.federated] [Coordinator] Step 3: Running simulation at ANL
17:00:01 [academy.federated] [ANL] Simulation complete: energy=-127.5
17:00:01 [academy.federated] [Coordinator] Step 4: Analyzing results at LBNL
17:00:01 [academy.federated] [LBNL] Analysis complete: confidence=0.94
17:00:01 [academy.federated] [Coordinator] Federated workflow complete
----------------------------------------------------------------------
FEDERATED WORKFLOW RESULTS
----------------------------------------------------------------------
Status: completed
Institutions: ANL, ORNL, LBNL

Summary:
  Data records used (ORNL):    1247
  Simulation energy (ANL):     -127.5
  Analysis confidence (LBNL):  0.94

Workflow steps:
  1. capability_discovery (all)
  2. data_query (ORNL)
  3. simulation (ANL)
  4. analysis (LBNL)
======================================================================
```

## Key Concepts

### 1. Institutional Agents

Each institution provides agents with specialized capabilities:

```python
class ANLComputeAgent(Agent):
    """Argonne's compute capabilities (Aurora, Polaris)."""
    institution = "ANL"

    @action
    async def run_simulation(self, parameters: dict) -> dict:
        # Run on Aurora exascale system
        ...

    @action
    async def get_capabilities(self) -> dict:
        return {
            "institution": self.institution,
            "capabilities": ["DFT", "MD", "ML_inference"],
            "resources": ["Aurora", "Polaris"],
        }
```

### 2. Coordinator Pattern

A coordinator orchestrates work across institutions:

```python
class FederatedCoordinator(Agent):
    @action
    async def register_agent(self, institution: str, agent: Handle) -> None:
        self._agents[institution] = agent

    @action
    async def execute_federated_workflow(self, task: str) -> dict:
        # 1. Query capabilities from all institutions
        # 2. Fetch data from ORNL
        # 3. Run simulation at ANL
        # 4. Analyze results at LBNL
        # 5. Compile and return results
```

### 3. Capability Discovery

Before executing, the coordinator queries what each institution offers:

```python
for institution, agent in self._agents.items():
    capabilities = await agent.get_capabilities()
    # Capabilities inform workflow decisions
```

## Production Deployment

| Aspect | Demo (this example) | Production |
|--------|---------------------|------------|
| **Exchange** | LocalExchangeFactory | GlobusComputeExchange, RedisExchange |
| **Identity** | None | Globus Auth, InCommon |
| **Policy** | None | Academy governance layer |
| **Network** | Same process | Institutional firewalls, VPNs |

## Real-World Applications

This pattern enables scenarios like:

1. **Materials discovery**: Combine DFT calculations (ANL), neutron data (ORNL), and ML analysis (LBNL)
2. **Climate modeling**: Distribute ensemble runs across multiple HPC centers
3. **Genomics pipelines**: Sequence at one lab, analyze at another, store at a third
4. **Light source experiments**: Collect data at synchrotrons, process on HPC, visualize remotely

## Security Considerations

In production deployments:

- **Authentication**: Use federated identity (Globus Auth, InCommon)
- **Authorization**: Define policies for who can call what
- **Data governance**: Track data provenance and access
- **Audit logging**: Record all cross-institutional operations

## Next Steps

- [Federated Agent Execution](/Capabilities/federated-agents/) - Full federated deployment guide
- [Governed Tool Use](/Capabilities/governed-tool-use/) - Policy enforcement
- [AgentsRemoteTools](/Capabilities/local-agents/AgentsRemoteTools/) - Remote tool pattern

## Requirements

- Python 3.10+
- academy-py
