#!/usr/bin/env python3
"""
Knowledge-Updating Research Agent

A simple agent that:
1. Searches arXiv for new papers on protein-ligand binding affinity
2. Uses Claude to triage relevance and extract binding data
3. Stores results in a SQLite database
4. Alerts on high-impact findings

Run periodically via cron, e.g.:
    0 2 * * * cd /path/to/knowledge-agent && python agent.py

Requires:
    pip install anthropic arxiv
    export ANTHROPIC_API_KEY=your-key
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path

import arxiv
import anthropic

# Configuration
CONFIG = {
    "search_query": "protein ligand binding affinity",
    "max_papers_per_run": 20,
    "lookback_days": 7,
    "db_path": "knowledge_base.db",
    "state_path": "agent_state.json",
    "records_before_retrain": 200,
    "high_impact_keywords": ["breakthrough", "novel", "significant", "state-of-the-art"],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent.log"),
    ],
)
log = logging.getLogger(__name__)


def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            abstract TEXT,
            published DATE,
            url TEXT,
            is_relevant BOOLEAN,
            extracted_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS binding_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            protein TEXT,
            ligand TEXT,
            affinity_value REAL,
            affinity_unit TEXT,
            method TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id) REFERENCES papers(id)
        )
    """)
    conn.commit()
    return conn


def load_state(state_path: str) -> dict:
    """Load agent state from JSON file."""
    if Path(state_path).exists():
        with open(state_path) as f:
            return json.load(f)
    return {
        "last_run": None,
        "total_papers_processed": 0,
        "total_records_extracted": 0,
        "records_since_retrain": 0,
        "model_version": 0,
    }


def save_state(state: dict, state_path: str):
    """Save agent state to JSON file."""
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def search_arxiv(query: str, max_results: int, since: datetime) -> list:
    """Search arXiv for recent papers."""
    log.info(f"Searching arXiv for: {query} (since {since.date()})")

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    for result in client.results(search):
        if result.published.replace(tzinfo=None) >= since:
            papers.append({
                "id": result.entry_id,
                "title": result.title,
                "authors": ", ".join(a.name for a in result.authors),
                "abstract": result.summary,
                "published": result.published.date(),
                "url": result.entry_id,
            })

    log.info(f"Found {len(papers)} papers since {since.date()}")
    return papers


def triage_paper(client: anthropic.Anthropic, paper: dict) -> dict:
    """Use Claude to determine if paper is relevant and extract data."""

    prompt = f"""Analyze this scientific paper about protein-ligand binding.

Title: {paper['title']}

Abstract: {paper['abstract']}

Tasks:
1. Is this paper relevant to protein-ligand binding affinity prediction or measurement? (yes/no)
2. If relevant, extract any binding affinity data mentioned (protein, ligand, affinity value, unit, method)
3. Is this a high-impact finding? (breakthrough methodology, significantly better results, novel approach)

Respond in JSON format:
{{
    "is_relevant": true/false,
    "relevance_reason": "brief explanation",
    "binding_data": [
        {{
            "protein": "protein name or target",
            "ligand": "ligand/compound name",
            "affinity_value": numeric value or null,
            "affinity_unit": "nM/uM/Ki/Kd/IC50/etc" or null,
            "method": "experimental/computational method" or null
        }}
    ],
    "is_high_impact": true/false,
    "impact_reason": "why this is significant" or null
}}

If no specific binding data is extractable, return an empty array for binding_data.
Only return the JSON, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        result = json.loads(response.content[0].text)
        return result
    except json.JSONDecodeError:
        log.warning(f"Failed to parse LLM response for paper: {paper['id']}")
        return {"is_relevant": False, "binding_data": [], "is_high_impact": False}


def store_paper(conn: sqlite3.Connection, paper: dict, triage_result: dict):
    """Store paper and extracted data in database."""

    # Store paper
    conn.execute("""
        INSERT OR REPLACE INTO papers (id, title, authors, abstract, published, url, is_relevant, extracted_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        paper["id"],
        paper["title"],
        paper["authors"],
        paper["abstract"],
        paper["published"],
        paper["url"],
        triage_result.get("is_relevant", False),
        json.dumps(triage_result),
    ))

    # Store binding data
    records_added = 0
    for data in triage_result.get("binding_data", []):
        if data.get("protein") or data.get("ligand"):
            conn.execute("""
                INSERT INTO binding_data (paper_id, protein, ligand, affinity_value, affinity_unit, method, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                paper["id"],
                data.get("protein"),
                data.get("ligand"),
                data.get("affinity_value"),
                data.get("affinity_unit"),
                data.get("method"),
                triage_result.get("relevance_reason"),
            ))
            records_added += 1

    conn.commit()
    return records_added


def check_retrain_needed(state: dict, config: dict) -> bool:
    """Check if we have enough new data to trigger retraining."""
    return state["records_since_retrain"] >= config["records_before_retrain"]


def trigger_retrain(state: dict):
    """
    Trigger model retraining.

    In a full implementation, this would:
    1. Request human approval
    2. Submit job to NERSC via Globus Compute
    3. Wait for completion
    4. Deploy new model

    For now, we just log and update state.
    """
    log.info("=" * 60)
    log.info("RETRAIN TRIGGERED")
    log.info(f"Records since last retrain: {state['records_since_retrain']}")
    log.info("In production, this would:")
    log.info("  1. Request human approval")
    log.info("  2. Submit retraining job to NERSC")
    log.info("  3. Deploy updated model")
    log.info("=" * 60)

    # Simulate: reset counter, increment model version
    state["records_since_retrain"] = 0
    state["model_version"] += 1


def alert_high_impact(paper: dict, triage_result: dict):
    """Alert user to high-impact finding."""
    log.info("=" * 60)
    log.info("HIGH-IMPACT FINDING DETECTED")
    log.info(f"Title: {paper['title']}")
    log.info(f"URL: {paper['url']}")
    log.info(f"Reason: {triage_result.get('impact_reason', 'N/A')}")
    log.info("=" * 60)

    # In production, this would send email/Slack notification


def run_agent():
    """Main agent loop."""
    log.info("=" * 60)
    log.info("Knowledge Agent Starting")
    log.info("=" * 60)

    # Initialize
    conn = init_db(CONFIG["db_path"])
    state = load_state(CONFIG["state_path"])
    client = anthropic.Anthropic()

    # Determine search window
    if state["last_run"]:
        since = datetime.fromisoformat(state["last_run"])
    else:
        since = datetime.now() - timedelta(days=CONFIG["lookback_days"])

    log.info(f"Last run: {state['last_run'] or 'never'}")
    log.info(f"Records since last retrain: {state['records_since_retrain']}")
    log.info(f"Model version: {state['model_version']}")

    # Search for papers
    papers = search_arxiv(
        CONFIG["search_query"],
        CONFIG["max_papers_per_run"],
        since,
    )

    # Process each paper
    relevant_count = 0
    records_added = 0
    high_impact_count = 0

    for paper in papers:
        # Check if already processed
        existing = conn.execute(
            "SELECT id FROM papers WHERE id = ?", (paper["id"],)
        ).fetchone()
        if existing:
            log.debug(f"Skipping already processed: {paper['id']}")
            continue

        log.info(f"Processing: {paper['title'][:60]}...")

        # Triage with LLM
        try:
            triage_result = triage_paper(client, paper)
        except Exception as e:
            log.error(f"Error triaging paper: {e}")
            continue

        # Store results
        added = store_paper(conn, paper, triage_result)
        records_added += added

        if triage_result.get("is_relevant"):
            relevant_count += 1
            log.info(f"  -> Relevant, extracted {added} records")

        if triage_result.get("is_high_impact"):
            high_impact_count += 1
            alert_high_impact(paper, triage_result)

    # Update state
    state["last_run"] = datetime.now().isoformat()
    state["total_papers_processed"] += len(papers)
    state["total_records_extracted"] += records_added
    state["records_since_retrain"] += records_added

    # Check if retrain needed
    if check_retrain_needed(state, CONFIG):
        trigger_retrain(state)

    save_state(state, CONFIG["state_path"])

    # Summary
    log.info("=" * 60)
    log.info("Run Complete")
    log.info(f"Papers processed: {len(papers)}")
    log.info(f"Relevant papers: {relevant_count}")
    log.info(f"Records extracted: {records_added}")
    log.info(f"High-impact findings: {high_impact_count}")
    log.info(f"Total records since retrain: {state['records_since_retrain']}")
    log.info("=" * 60)

    conn.close()


if __name__ == "__main__":
    run_agent()
