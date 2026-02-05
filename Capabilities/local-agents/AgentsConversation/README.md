# Conversational Agent with Memory

An interactive agent demonstrating memory patterns for stateful conversations.

**Code:** [View on GitHub](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsConversation)

## What It Does

Demonstrates three memory patterns:

1. **Short-term memory**: Recent conversation history (sliding window)
2. **Long-term memory**: Persistent facts extracted from conversation
3. **Memory compression**: Summarize old messages to manage context length

## Running the Example

```bash
cd Capabilities/local-agents/AgentsConversation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

With initial context:

```bash
python main.py --context "I'm researching copper catalysts"
```

Save/load memory across sessions:

```bash
python main.py --save memory.json
python main.py --load memory.json
```

## LLM Configuration

Supports OpenAI, FIRST (HPC inference), Ollama (local), or mock mode.

See [LLM Configuration](/Capabilities/local-agents/llm-configuration/) for details.

## Memory Architecture

<img src="/Capabilities/Assets/conversation-memory.svg" alt="Conversation memory: short-term messages, long-term facts, sliding window, and summary compression" style="max-width: 400px; margin: 1rem 0;">

## Tools

| Tool | Description |
|------|-------------|
| `remember_fact` | Store a fact in long-term memory |
| `recall_fact` | Retrieve a stored fact |
| `list_known_facts` | List all stored facts |
| `get_conversation_summary` | Get conversation stats and summary |
| `search_literature` | Search scientific papers (demo) |
| `calculate_property` | Calculate molecular property (demo) |

## Key Patterns

**Fact Extraction**: Agent proactively stores important information:
```python
@tool
def remember_fact(key: str, value: str) -> str:
    MEMORY.add_fact(key, value)
    return f"Remembered: {key} = {value}"
```

**Context Window Management**: Old messages compress into summary:
```python
def _compress_old_messages(self):
    old_messages = self.messages[:-keep_count]
    self.messages = self.messages[-keep_count:]
    self.summary += summarize(old_messages)
```

**Persistent Memory**: Save/load between sessions:
```python
memory.save(Path("memory.json"))
memory = ConversationMemory.load(Path("memory.json"))
```

## Requirements

- Python 3.10+
- LangGraph 1.0+
- OpenAI API key, FIRST token, Ollama, or run in mock mode
