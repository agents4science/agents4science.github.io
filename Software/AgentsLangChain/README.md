
# AgentsLangChain

A LangChain-based reimplementation of the **Agents4Science** classroom framework.
This version replaces the custom agent orchestration with LangChain's
`ChatOpenAI`, `tool`, and `AgentExecutor` primitives.

## Roles
| Role | Function |
|------|----------|
| **Scout** | Detect anomalies, propose opportunities |
| **Planner** | Design workflows, allocate resources |
| **Operator** | Execute safely |
| **Analyst** | Summarize, quantify uncertainty (uses `analyze_dataset` tool) |
| **Archivist** | Record provenance |

## Run

```bash
cd Software/AgentsLangChain
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=<your_api_key>
python main_langchain.py
```
