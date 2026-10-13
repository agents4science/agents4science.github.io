#!/usr/bin/env python3
"""
Knowledge-Updating Research Agent

A simple agent that:
1. Searches arXiv for new papers on protein-ligand binding affinity
2. Uses an LLM to triage relevance and extract binding data
3. Stores results in a SQLite database
4. Alerts on high-impact findings

Run periodically via cron, e.g.:
    0 2 * * * cd /path/to/knowledge-agent && python agent.py

Requires:
    pip install anthropic arxiv openai

LLM fallback chain (uses first available):
    1. OpenAI (if OPENAI_API_KEY is set)
    2. Anthropic (if ANTHROPIC_API_KEY is set)
    3. Argonne ALCF (if Globus tokens available)
    4. Mock (for testing)

Options:
    --reset    Clear database and state, start fresh
"""

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path

import arxiv

from llm_fallback import create_llm, LLMProvider

# Configuration
CONFIG = {
    "search_query": "autonomous AI agent scientific discovery",
    "max_papers_per_run": 100,
    "lookback_days": 30,
    "db_path": "knowledge_base.db",
    "json_path": "papers.json",
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
        CREATE TABLE IF NOT EXISTS agent_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            agent_name TEXT,
            architecture TEXT,
            domain TEXT,
            capabilities TEXT,
            llm_used TEXT,
            benchmark_results TEXT,
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


def load_papers_json(json_path: str) -> dict:
    """Load papers from JSON file."""
    if Path(json_path).exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


def save_papers_json(papers: dict, json_path: str):
    """Save papers to JSON file."""
    with open(json_path, "w") as f:
        json.dump(papers, f, indent=2, default=str)


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


def triage_paper(llm: LLMProvider, paper: dict) -> dict:
    """Use LLM to determine if paper is relevant and extract data."""

    system = "You are a scientific assistant that analyzes research papers about AI agents for scientific discovery. Always respond with valid JSON only."

    user = f"""Analyze this scientific paper about AI agents for scientific discovery.

Title: {paper['title']}

Abstract: {paper['abstract']}

Tasks:
1. Is this paper relevant to AI/LLM agents for scientific discovery, autonomous experimentation, or AI-driven research? (yes/no)
2. If relevant, extract information about the agent system described
3. Is this a high-impact finding? (novel architecture, significant capability, breakthrough results)

Respond in JSON format:
{{
    "is_relevant": true/false,
    "relevance_reason": "brief explanation",
    "agent_data": [
        {{
            "agent_name": "name of the agent/system" or null,
            "architecture": "single-agent/multi-agent/hierarchical/etc" or null,
            "domain": "chemistry/biology/materials/physics/general/etc" or null,
            "capabilities": "tool use, planning, reasoning, experimentation, etc" or null,
            "llm_used": "GPT-4/Claude/Llama/etc" or null,
            "benchmark_results": "any quantitative results mentioned" or null
        }}
    ],
    "is_high_impact": true/false,
    "impact_reason": "why this is significant" or null
}}

If no specific agent data is extractable, return an empty array for agent_data.
Only return the JSON, no other text."""

    response = llm.complete(system=system, user=user, max_tokens=1024)

    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        log.warning(f"Failed to parse LLM response for paper: {paper['id']}")
        log.debug(f"Raw response: {response[:200]}")
        return {"is_relevant": False, "agent_data": [], "is_high_impact": False}


def store_paper(conn: sqlite3.Connection, papers_json: dict, paper: dict, triage_result: dict):
    """Store paper and extracted data in both SQLite and JSON."""

    # Store in SQLite
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

    # Store agent data in SQLite
    records_added = 0
    for data in triage_result.get("agent_data", []):
        if data.get("agent_name") or data.get("architecture") or data.get("domain"):
            # Convert any dict/list values to JSON strings for SQLite
            def to_str(val):
                if isinstance(val, (dict, list)):
                    return json.dumps(val)
                return val

            conn.execute("""
                INSERT INTO agent_data (paper_id, agent_name, architecture, domain, capabilities, llm_used, benchmark_results, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper["id"],
                to_str(data.get("agent_name")),
                to_str(data.get("architecture")),
                to_str(data.get("domain")),
                to_str(data.get("capabilities")),
                to_str(data.get("llm_used")),
                to_str(data.get("benchmark_results")),
                triage_result.get("relevance_reason"),
            ))
            records_added += 1

    conn.commit()

    # Store in JSON (keyed by paper ID)
    papers_json[paper["id"]] = {
        "title": paper["title"],
        "authors": paper["authors"],
        "abstract": paper["abstract"],
        "published": str(paper["published"]),
        "url": paper["url"],
        "triage": triage_result,
    }

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
    papers_json = load_papers_json(CONFIG["json_path"])
    llm = create_llm(verbose=False)
    log.info(f"Using LLM: {llm.name}")

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
            triage_result = triage_paper(llm, paper)
        except Exception as e:
            log.error(f"Error triaging paper: {e}")
            continue

        # Store results
        added = store_paper(conn, papers_json, paper, triage_result)
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
    save_papers_json(papers_json, CONFIG["json_path"])

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


def reset_agent():
    """Clear database and state files."""
    log.info("Resetting agent...")
    for f in [CONFIG["db_path"], CONFIG["json_path"], CONFIG["state_path"], "agent.log"]:
        if Path(f).exists():
            Path(f).unlink()
            log.info(f"  Deleted {f}")
    log.info("Reset complete")


if __name__ == "__main__":
    if "--reset" in sys.argv:
        reset_agent()
    run_agent()
