# Knowledge-Updating Research Agent

A simple implementation of the knowledge-updating research agent from the [DOE Agent Requirements](../../DOE_Agent_Requirements/) document.

## What it does

1. **Searches arXiv** for recent papers on protein-ligand binding affinity
2. **Triages relevance** using Claude to determine if papers are relevant
3. **Extracts binding data** (protein, ligand, affinity values) from abstracts
4. **Stores in SQLite** for querying and analysis
5. **Alerts on high-impact** findings (logs to console; extend for email/Slack)
6. **Tracks retraining threshold** (stubs out the actual NERSC job submission)

## Setup

```bash
cd examples/knowledge-agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
```

## Run

```bash
python agent.py
```

## Run periodically (cron)

```bash
# Run daily at 2am
0 2 * * * cd /path/to/knowledge-agent && python agent.py >> agent.log 2>&1
```

## Files created

- `knowledge_base.db` — SQLite database with papers and extracted binding data
- `agent_state.json` — Agent state (last run, record counts, model version)
- `agent.log` — Log file

## Query the knowledge base

```bash
sqlite3 knowledge_base.db

-- Recent relevant papers
SELECT title, published FROM papers WHERE is_relevant = 1 ORDER BY published DESC LIMIT 10;

-- Extracted binding data
SELECT protein, ligand, affinity_value, affinity_unit, method
FROM binding_data
ORDER BY created_at DESC LIMIT 20;

-- Count by protein
SELECT protein, COUNT(*) as count FROM binding_data GROUP BY protein ORDER BY count DESC;
```

## Configuration

Edit the `CONFIG` dict in `agent.py`:

```python
CONFIG = {
    "search_query": "protein ligand binding affinity",  # arXiv search terms
    "max_papers_per_run": 20,                           # papers to process per run
    "lookback_days": 7,                                 # initial lookback window
    "records_before_retrain": 200,                      # trigger retrain threshold
    ...
}
```

## Extending

**Add more sources** (PubMed, bioRxiv):
- PubMed: Use `biopython` with `Entrez`
- bioRxiv: Use their RSS feed or API

**Real alerts**:
- Email: Use `smtplib` or SendGrid
- Slack: Use `slack_sdk`

**Real retraining**:
- Use Globus Compute to submit jobs to NERSC
- Add human approval flow (could be a simple prompt or web interface)

## Limitations

This is a demonstration. A production version would need:

- Better error handling and retries
- Rate limiting for APIs
- More sophisticated extraction (full-text, not just abstracts)
- Actual model retraining integration
- Human approval workflow
- Proper secrets management
