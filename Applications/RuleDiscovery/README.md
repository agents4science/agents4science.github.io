# Rule Discovery

Multi-agent simulation where LLM-powered agents discover hidden rules governing a simulated world through experimentation and natural language communication.

## Problem

A simulated world contains objects with properties and hidden rules that determine experimental outcomes.

**Objects have four properties:**
- **Color**: red, blue, green, yellow
- **Size**: small, medium, large
- **Material**: metal, wood, glass, rubber
- **Shape**: sphere, cube, pyramid, cylinder

**Example hidden rules:**
```
- Metal objects conduct electricity
- Glass objects shatter when dropped
- Blue objects float in water
- Wood objects burn when exposed to fire
```

**Agents can run experiments like:**
```
"Drop the small red metal sphere into water"
"Apply electricity to the large blue glass cube"
"Expose the medium green wood pyramid to fire"
```

## Agent Architecture

Each LLM-powered agent runs an asynchronous loop:

```
┌─────────────────────────────────────────────────────────────┐
│  1. PROPOSE EXPERIMENT (LLM)                                │
│     → "Drop the small red metal sphere into water"          │
├─────────────────────────────────────────────────────────────┤
│  2. RUN EXPERIMENT (World)                                  │
│     → "The small red metal sphere sinks to the bottom"      │
├─────────────────────────────────────────────────────────────┤
│  3. UPDATE HYPOTHESES (LLM)                                 │
│     → "Metal objects sink in water" (75% confidence)        │
├─────────────────────────────────────────────────────────────┤
│  4. SHARE WITH PEERS (probabilistic)                        │
│     → Send top hypotheses + recent findings to random peer  │
├─────────────────────────────────────────────────────────────┤
│  5. INTEGRATE PEER INFO (LLM)                               │
│     → Consider peer hypotheses, refine own understanding    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -r requirements.txt
```

## LLM Providers

Supports multiple backends via environment variables or command-line arguments:

| Provider | Use Case | Configuration |
|----------|----------|---------------|
| `mock` | Testing without API calls | Default, no config needed |
| `anthropic` | Claude API | `LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=...` |
| `openai_compatible` | Argonne, vLLM, TGI, etc. | `LLM_PROVIDER=openai_compatible LLM_BASE_URL=...` |

## Usage

### Mock Mode (Testing)

```bash
python hidden_rule_discovery.py --agents 4 --steps 50
```

### With Anthropic Claude

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-key-here
export LLM_MODEL=claude-3-haiku-20240307  # or claude-3-sonnet, claude-3-opus
python hidden_rule_discovery.py --agents 4 --steps 50
```

### With Argonne/OpenAI-Compatible Endpoint

```bash
export LLM_PROVIDER=openai_compatible
export LLM_BASE_URL=https://your-inference-endpoint/v1
export LLM_MODEL=meta-llama/Llama-3-70b-chat-hf
python hidden_rule_discovery.py --agents 4 --steps 50
```

### Command Line Options

```
python hidden_rule_discovery.py --help

Options:
  --provider, -p    LLM provider: mock, anthropic, openai_compatible
  --model, -m       Model name (provider-specific)
  --agents, -a      Number of agents (default: 4)
  --steps, -s       Number of steps (default: 50)
  --difficulty, -d  easy | medium | hard
  --comm-prob, -c   Communication probability (default: 0.3)
  --no-comm         Disable communication between agents
  --seed            Random seed for reproducibility
```

## Difficulty Levels

| Level | Rules | Description |
|-------|-------|-------------|
| **easy** | 3 | Simple, non-conflicting rules (e.g., material-based) |
| **medium** | 5 | Property interactions (e.g., color + experiment type) |
| **hard** | 8 | Complex interactions across multiple properties |

### Example Rules by Difficulty

**Easy (3 rules):**
```
- Metal objects conduct electricity
- Glass objects shatter when dropped
- Rubber objects bounce when thrown
```

**Medium (5 rules):**
```
- Metal objects conduct electricity
- Wood objects burn when exposed to fire
- Blue objects float in water
- Large objects sink in water
- Glass objects crack in the freezer
```

**Hard (8 rules):**
```
- Metal objects conduct electricity
- Metal objects heat up in sunlight
- Wood objects burn in fire
- Wood objects float in water
- Red objects are fireproof
- Small objects fly far when thrown
- Spheres roll when dropped on the floor
- Pyramids tip over on a scale
```

## Sample Results

### Mock Mode Run (4 agents, 20 steps, easy difficulty)

```
======================================================================
Hidden Rule Discovery with LLM Agents
======================================================================

LLM Provider: MockLLM
Difficulty: easy
Number of hidden rules: 3
Number of agents: 4
Communication probability: 0.3

[Hidden rules - for reference only]
- Metal objects conduct electricity
- Glass objects shatter when dropped
- Rubber objects bounce when thrown

--- Step 10 ---

Agent 0:
  Observations: 20
  Messages: sent=4, recv=3
  Top hypotheses:
    - Metal objects conduct electricity (85%)
    - Glass objects shatter when dropped (80%)

Agent 1:
  Observations: 20
  Messages: sent=3, recv=4
  Top hypotheses:
    - Rubber objects bounce when thrown (75%)
    - Metal objects conduct electricity (70%)

--- Step 20 ---

Agent 0:
  Observations: 40
  Messages: sent=8, recv=7

Agent 1:
  Observations: 40
  Messages: sent=7, recv=8

======================================================================
FINAL EVALUATION
======================================================================

Overall discovery score: 72/100

Evaluation details:
- Metal objects conduct electricity: fully discovered
- Glass objects shatter when dropped: partially discovered
- Rubber objects bounce when thrown: fully discovered

Total experiments run: 160
```

### Expected Results with Real LLM

With a real LLM (Claude or Llama), agents will:
1. **Design systematic experiments** - varying one property at a time
2. **Form precise hypotheses** - based on observed patterns
3. **Share discoveries** - accelerating collective learning
4. **Refine understanding** - incorporating peer findings

Typical discovery scores:
- **Easy difficulty**: 70-90% with communication, 50-70% without
- **Medium difficulty**: 50-75% with communication, 30-50% without
- **Hard difficulty**: 30-60% with communication, 20-40% without

## Evaluation

Performance is measured by comparing discovered hypotheses against ground-truth rules using LLM-based semantic matching:

1. Each true rule is assessed as: **fully** / **partially** / **missed**
2. Overall score: 0-100 based on discovery completeness
3. Explanation provided for the scoring rationale

## Files

| File | Description |
|------|-------------|
| `hidden_rule_discovery.py` | Main LLM agent simulation |
| `llm_providers.py` | Provider abstraction (Mock, Anthropic, OpenAI-compatible) |
| `requirements.txt` | Python dependencies |

## Extending

### Adding a New LLM Provider

1. Create a new class inheriting from `LLMProvider` in `llm_providers.py`
2. Implement the `complete(system, user, max_tokens)` method
3. Add to the `create_llm_provider()` factory function

### Creating Custom Worlds

Modify the `create_world()` function in `hidden_rule_discovery.py` to define new rule sets:

```python
rules = [
    Rule(
        condition="material == metal",
        experiment_type="electricity",
        outcome="conducts electricity",
        natural_language="Metal objects conduct electricity"
    ),
    # Add more rules...
]
return HiddenWorld(rules=rules, seed=seed)
```
