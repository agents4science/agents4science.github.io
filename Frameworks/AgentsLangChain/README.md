# AgentsLangChain

A LangChain-based simplified reimplementation of AgentsExample, using LangChain's `ChatOpenAI`, `tool`, and `AgentExecutor` primitives. This version does not include the dashboard UI, Argonne inference service support, or multiple tools.

**Code:** [github.com/agents4science/agents4science.github.io/tree/main/Frameworks/AgentsLangChain](https://github.com/agents4science/agents4science.github.io/tree/main/Frameworks/AgentsLangChain)

## Example

The example runs a pipeline of five specialized agents working on a scientific goal: "Find catalysts that improve CO₂ conversion at room temperature." Each agent processes the goal and passes its output to the next agent in the chain, demonstrating a simple multi-agent workflow for scientific discovery.

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
cd Frameworks/AgentsLangChain
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=<your_api_key>
python main_langchain.py
```
