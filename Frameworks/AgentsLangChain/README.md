
# AgentsLangChain

A LangChain-based simplified reimplementation of the AgentsExample, replacing 
the custom agent orchestration with LangChain's
`ChatOpenAI`, `tool`, and `AgentExecutor` primitives.
Does not provide dashboard, does not use Argonne inference service, and only has one tool.

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
