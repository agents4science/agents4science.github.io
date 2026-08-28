#!/usr/bin/env python3
"""MCP server exposing the instrument capability graph as agent tools.

Register in Claude Code:
  claude mcp add capability-graph -- python3 /Users/ian/AAA/Code/GM_Instruments/capability_graph/queries/mcp_server.py
Then ask, e.g.: "Which instrument can measure trace element distribution in plant roots?"
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from matchmaker import load_records, search           # keyword+synonym scorer
import ir_matcher                                      # hard-constraint pipeline

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("capability-graph")

@mcp.tool()
def find_instruments(query: str, include_unavailable: bool = False, top: int = 5) -> str:
    """Rank instruments for a free-text measurement need (e.g. 'oxidation state
    of a catalyst under reaction gas'). Searches 125 records across ALS, APS,
    and PoLARIS. Returns ranked candidates with matched terms and status."""
    results, _ = search(query, include_all=include_unavailable, top=top)
    out = []
    for s, r, matched in results:
        out.append({"id": r["id"], "name": r["name"], "facility": r["facility"],
                    "status": r.get("status"), "score": round(s, 1),
                    "record_depth": r.get("record_depth"),
                    "techniques": [t.get("acronym") or t["name"] for t in r["techniques"]],
                    "matched": {k: sorted(v) for k, v in matched.items()}})
    return json.dumps(out, indent=1)

@mcp.tool()
def get_instrument(instrument_id: str) -> str:
    """Full capability record for one instrument (e.g. 'als-9.3.2',
    'aps-13-id-e', 'polaris-dsc2500'), including sample constraints,
    sample environments, throughput, automation interfaces, demonstrated
    applications, and per-source provenance."""
    for r in load_records():
        if r["id"] == instrument_id:
            return json.dumps(r, indent=1)
    return json.dumps({"error": f"no record with id '{instrument_id}'",
                       "hint": "use find_instruments or list_instruments first"})

@mcp.tool()
def match_measurement_request(ir_json: str, show_excluded: bool = False) -> str:
    """Match a structured measurement-request IR (JSON conforming to
    measurement_request.schema.json) against the catalog using the
    deterministic hard-constraint filter. Returns candidates with
    per-constraint PASS/FAIL/UNKNOWN verdicts, near-misses, and suggested
    clarifying questions. Missing data yields UNKNOWN, never FAIL."""
    ir = json.loads(ir_json)
    records = load_records()
    candidates, excluded = [], []
    for r in records:
        results = {name: fn(r, ir) for name, fn in ir_matcher.CHECKS}
        fails = [(n, reason) for n, (v, reason) in results.items() if v == ir_matcher.FAIL]
        entry = {"id": r["id"], "name": r["name"], "status": r.get("status"),
                 "checks": {n: {"verdict": v, "reason": reason} for n, (v, reason) in results.items()}}
        if fails:
            entry["excluded_because"] = [f"{n}: {reason}" for n, reason in fails]
            excluded.append(entry)
        else:
            entry["score"] = round(ir_matcher.soft_score(r, ir, results), 1)
            candidates.append(entry)
    candidates.sort(key=lambda c: -c["score"])
    near = [e for e in excluded if len(e["excluded_because"]) == 1][:6]
    resp = {"candidates": candidates[:5], "near_misses": near,
            "clarifying_questions": ir_matcher.clarifications(ir, [{"record": r} for r in records])}
    if show_excluded:
        resp["excluded"] = excluded
    return json.dumps(resp, indent=1)

@mcp.tool()
def catalog_stats() -> str:
    """Catalog summary: record counts by facility, depth, status, and class."""
    from collections import Counter
    rs = load_records()
    return json.dumps({
        "total": len(rs),
        "by_facility": dict(Counter(r["facility"] for r in rs)),
        "by_status": dict(Counter(r.get("status", "unknown") for r in rs)),
        "by_depth": dict(Counter(r.get("record_depth", "?") for r in rs)),
        "by_class": dict(Counter(r["instrument_class"] for r in rs))}, indent=1)

if __name__ == "__main__":
    mcp.run()
