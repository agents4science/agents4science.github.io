# Knowledge-Updating Research Agent

A simple agent that monitors arXiv for new papers on a topic of interest, uses an LLM to triage relevance and extract structured data, and stores results for analysis.

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   arXiv     │────▶│  LLM Triage │────▶│   Store     │────▶│   Alert     │
│   Search    │     │  & Extract  │     │  Results    │     │  High-Impact│
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

1. **Search**: Queries arXiv for recent papers matching your search terms
2. **Triage**: Sends each paper's title/abstract to an LLM to determine relevance
3. **Extract**: LLM extracts structured data (agent name, architecture, domain, etc.)
4. **Store**: Saves to both SQLite (for queries) and JSON (for browsing)
5. **Alert**: Logs high-impact findings (extend for email/Slack)

## Setup

```bash
cd examples/knowledge-agent
pip install -r requirements.txt
```

### LLM Configuration

The agent uses an automatic fallback chain—it tries providers in order until one works:

| Priority | Provider | How to Enable |
|----------|----------|---------------|
| 1 | OpenAI | Set `OPENAI_API_KEY` |
| 2 | Anthropic | Set `ANTHROPIC_API_KEY` |
| 3 | Argonne ALCF | Globus tokens at `~/.globus/app/.../inference_app/tokens.json` |
| 4 | Mock | Always available (returns dummy responses) |

Example:
```bash
export ANTHROPIC_API_KEY=your-key-here
# or just use Argonne if you have Globus tokens set up
```

## Usage

### Run once
```bash
python agent.py
```

### Reset and run fresh
```bash
python agent.py --reset
```

### Run periodically (cron)
```bash
# Run daily at 2am
0 2 * * * cd /path/to/knowledge-agent && python agent.py >> agent.log 2>&1
```

## Output Files

| File | Format | Description |
|------|--------|-------------|
| `papers.json` | JSON | All papers with extracted data (human-readable) |
| `knowledge_base.db` | SQLite | Same data in queryable database |
| `agent_state.json` | JSON | Agent state (last run, counts) |
| `agent.log` | Text | Run logs |

## Exploring Results

### JSON (simple browsing)
```bash
# Pretty print
cat papers.json | python -m json.tool

# With jq
jq '.[] | select(.triage.is_relevant) | .title' papers.json
jq '.[] | select(.triage.is_high_impact) | {title, reason: .triage.impact_reason}' papers.json
```

### SQLite (queries)
```bash
sqlite3 knowledge_base.db

-- Recent relevant papers
SELECT title, published FROM papers
WHERE is_relevant = 1
ORDER BY published DESC LIMIT 10;

-- Extracted agent data
SELECT agent_name, architecture, domain, capabilities
FROM agent_data;

-- Count by domain
SELECT domain, COUNT(*) as count
FROM agent_data
GROUP BY domain
ORDER BY count DESC;

-- High-impact papers
SELECT title, json_extract(extracted_data, '$.impact_reason') as reason
FROM papers
WHERE json_extract(extracted_data, '$.is_high_impact') = 1;
```

### Python
```python
import json

with open('papers.json') as f:
    papers = json.load(f)

# Relevant papers
relevant = [p for p in papers.values() if p['triage'].get('is_relevant')]
print(f"Found {len(relevant)} relevant papers")

# High-impact
for p in papers.values():
    if p['triage'].get('is_high_impact'):
        print(f"- {p['title']}")
        print(f"  {p['triage'].get('impact_reason')}\n")
```

## Configuration

Edit the `CONFIG` dict in `agent.py`:

```python
CONFIG = {
    "search_query": "autonomous AI agent scientific discovery",  # arXiv search
    "max_papers_per_run": 100,      # papers to fetch per run
    "lookback_days": 30,            # initial lookback window
    "records_before_retrain": 200,  # (placeholder for future use)
}
```

## Customizing for Different Topics

To track a different research area:

1. Change `search_query` in CONFIG
2. Update the triage prompt in `triage_paper()` to extract relevant fields
3. Update the `agent_data` table schema in `init_db()` if needed
4. Run with `--reset` to start fresh

## Architecture

```
agent.py           - Main agent logic
llm_fallback.py    - LLM provider abstraction with fallback chain
requirements.txt   - Python dependencies
README.md          - This file
```

The `llm_fallback.py` module provides a unified interface across LLM providers:
- `create_llm()` - Factory that returns the best available provider
- `LLMProvider.complete(system, user, max_tokens)` - Simple completion API

## Limitations

This is a demonstration. A production version would need:

- Rate limiting for arXiv and LLM APIs
- Better error handling and retries
- Full-text extraction (not just abstracts)
- Email/Slack notifications for high-impact findings
- Web UI for browsing results
