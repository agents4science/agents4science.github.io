# LLM Configuration

The examples in this directory support multiple LLM backends: OpenAI, FIRST (HPC inference), Ollama (local), or mock mode (no setup required).

## Supported Modes

| Mode | Environment Variable | Description |
|------|---------------------|-------------|
| **OpenAI** | `OPENAI_API_KEY` | Uses OpenAI (gpt-4o-mini by default) |
| **FIRST** | `FIRST_API_KEY` | Uses FIRST HPC inference service |
| **Ollama** | `OLLAMA_MODEL` | Uses Ollama for local LLM inference |
| **Mock** | (none) | Demonstrates patterns with hardcoded responses |

Precedence when multiple variables are set: OpenAI > FIRST > Ollama > Mock.

## Configuration

### OpenAI

```bash
export OPENAI_API_KEY=<your_key>
python main.py
```

### FIRST (HPC Inference)

```bash
export FIRST_API_KEY=<your_token>
export FIRST_API_BASE=https://your-first-endpoint/v1
export FIRST_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
python main.py
```

### Ollama (Local LLM)

[Ollama](https://ollama.com/) runs LLMs locally on your machine.

```bash
# Install Ollama and pull a model
ollama pull llama3.2

# Run with Ollama
export OLLAMA_MODEL=llama3.2
python main.py
```

Optional: Set `OLLAMA_HOST` if Ollama is running on a different host (default: `http://localhost:11434`).

### Mock Mode (No Setup Required)

```bash
python main.py
```

Mock mode runs without any API key, showing realistic example outputs to demonstrate the patterns.

## Mode Detection Output

When you run an example, it prints which mode was selected and why:

```
============================================================
LLM Mode: OpenAI (gpt-4o-mini)
  Reason: OPENAI_API_KEY found in environment
============================================================
```

Or in mock mode:

```
============================================================
LLM Mode: Mock
  Reason: No API key or OLLAMA_MODEL found; using hardcoded responses
============================================================
```

## Using Argonne's FIRST Service

Argonne National Laboratory provides access to FIRST through the ALCF (Argonne Leadership Computing Facility).

### Getting Access

1. **Get an ALCF account** if you don't have one
2. **Obtain an API token** following the instructions at:
   [docs.alcf.anl.gov/services/inference-endpoints/#api-access](https://docs.alcf.anl.gov/services/inference-endpoints/#api-access)

### Argonne Configuration

```bash
export FIRST_API_KEY=<your_token>
export FIRST_API_BASE=https://inference-api.alcf.anl.gov/resource_server/metis/api/v1
export FIRST_MODEL=gpt-oss-120b
python main.py
```

### Available Models at Argonne

Check the [ALCF documentation](https://docs.alcf.anl.gov/services/inference-endpoints/) for the current list of available models. Common options include:

- `gpt-oss-120b` - Large general-purpose model
- `meta-llama/Meta-Llama-3.1-70B-Instruct` - Llama 3.1 70B

## Model Recommendations for Tool Calling

Not all models handle tool calling equally well. Based on testing:

### Recommended Models

| Use Case | Recommended Models |
|----------|-------------------|
| **Simple tools** (1-2 tools, clear inputs) | Any model, including `llama3.2` (3B) |
| **Multiple tools** (3+ tools) | `llama3.2:70b`, `mistral`, `gpt-4o-mini` |
| **Complex workflows** (multi-step, conditional) | `gpt-4o`, `gpt-4o-mini`, `llama3.1:70b` |

### Known Limitations with Smaller Models

When using smaller local models like `llama3.2` (3B parameters), you may observe:

- **Code generation instead of tool calls**: Model outputs Python code describing what it would do, rather than invoking the tool
- **Parameter hallucination**: Model invents parameter values instead of using provided options
- **Incomplete tool sequences**: Model describes remaining steps instead of executing them

**Example of correct tool call:**
```
Agent calls: calculate({'expression': '347 * 892'})
Tool result: 309524
```

**Example of problematic behavior (smaller models):**
```
Agent: I'll calculate this using the following code:
```python
result = 347 * 892
print(result)  # 309524
```
```

### Recommendations

1. **Start with mock mode** to understand the expected flow
2. **Use OpenAI or larger models** for production or complex examples
3. **Ollama with small models** works well for:
   - Simple calculator-style tools
   - Single-tool scenarios
   - Learning and experimentation
4. **Increase model size** if you see code generation instead of tool calls
