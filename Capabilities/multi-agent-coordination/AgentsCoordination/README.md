# Multi-Agent Coordination

Multiple agents sharing budget, state, and policies.

**Code:** [View on GitHub](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/multi-agent-coordination/AgentsCoordination)

## What It Does

Demonstrates coordination patterns for concurrent agents:

1. **Shared budget**: All agents draw from a common resource pool
2. **Shared blackboard**: Agents post findings for others to read
3. **Task claiming**: Prevent duplicate work across agents
4. **Per-agent quotas**: Limit operations per agent
5. **Event logging**: Track all coordination activity

## Running the Example

```bash
cd Capabilities/multi-agent-coordination/AgentsCoordination
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Custom configuration:

```bash
python main.py --agents 5 --budget 200 --tasks 10
```

## LLM Configuration

Supports OpenAI, FIRST (HPC inference), Ollama (local), or mock mode.

See [LLM Configuration](/Capabilities/local-agents/llm-configuration/) for details.

## Coordination Architecture

<img src="/Capabilities/Assets/coordination-hub.svg" alt="Coordination hub with shared budget, blackboard, and agent quotas" style="max-width: 500px; margin: 1rem 0;">

## Key Patterns

**Thread-safe shared budget:**
```python
@dataclass
class SharedBudget:
    total: float
    spent: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def request(self, amount: float, agent_id: str) -> tuple[bool, str]:
        with self._lock:
            if self.spent + amount > self.total:
                return False, "Insufficient budget"
            self.spent += amount
            return True, f"Allocated ${amount}"
```

**Task claiming to prevent duplicate work:**
```python
def claim_task(self, task_id: str, agent_id: str) -> tuple[bool, str]:
    with self._lock:
        if task_id in self._claims:
            return False, f"Already claimed by {self._claims[task_id]}"
        self._claims[task_id] = agent_id
        return True, "Claimed"
```

**Blackboard for sharing findings:**
```python
def post(self, key: str, value: any, agent_id: str):
    with self._lock:
        self._data[key] = {
            "value": value,
            "posted_by": agent_id,
            "timestamp": datetime.now().isoformat()
        }
```

## Tools

| Tool | Description |
|------|-------------|
| `request_budget` | Request allocation from shared budget |
| `post_finding` | Share a finding on the blackboard |
| `read_findings` | Read all findings from blackboard |
| `claim_task` | Claim a task (prevents duplicates) |
| `release_task` | Release a claimed task |
| `check_budget_status` | Check shared budget |
| `get_my_quota` | Check this agent's quota |

## Example Output

```
============================================================
MULTI-AGENT COORDINATION
============================================================
Agents: 3
Total budget: $100.00
Tasks: 6
============================================================

[Agent_1] Task 'Analyze' claimed by Agent_1
[Agent_1] Allocated $10.00 to Agent_1. Remaining: $90.00
[Agent_1] Posted 'Agent_1_result' to blackboard

[Agent_2] Task 'Run' claimed by Agent_2
[Agent_2] Allocated $10.00 to Agent_2. Remaining: $80.00
[Agent_2] Posted 'Agent_2_result' to blackboard

============================================================
COORDINATION SUMMARY
============================================================

Budget: $60.00 / $100.00 (60.0% used)

Shared findings (6):
  - Agent_1_result: Result from Agent_1: analysis complete...
  - Agent_2_result: Result from Agent_2: analysis complete...

Recent coordination events:
  [Agent_1] budget_request: Allocated $10.00
  [Agent_2] post_finding: Posted 'Agent_2_result'
```

## Production Considerations

This example uses in-memory coordination. Production systems should:

- **Distributed state**: Use Redis, etcd, or database for shared state
- **Persistent budget**: Track spending in durable storage
- **Failure handling**: Release claimed tasks if agent crashes
- **Priority queues**: Handle high-priority tasks first
- **Load balancing**: Distribute work based on agent capacity

## Requirements

- Python 3.10+
- LangGraph 1.0+
- OpenAI API key, FIRST token, Ollama, or run in mock mode
