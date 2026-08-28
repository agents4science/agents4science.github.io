# IFM: Instrument Capability Graph

A machine-readable catalog of scientific-instrument capabilities, plus
tools that resolve "I want to measure X" to ranked instruments. This is
the coarse-fidelity bootstrap of an **instrument forward model (IFM)**:

    I(instrument, protocol, sample) -> (data product, resolution, uncertainty, cost, time)

Selecting an instrument, operating it, and designing a new one are three
inversions of this one model: over the instrument catalog, over protocol
space, and over component graphs, respectively.

## Contents

- `schema/instrument_capability.schema.json` - capability record schema
  (v0.2). Key design decisions: `measurables` (WHAT you learn) separated
  from `techniques` (HOW); `demonstrated_applications` mined from science
  highlights, not spec sheets; per-source provenance on every record;
  `throughput`/`automation`/`campaign_roles` fields for autonomous-lab
  campaign planning.
- `schema/measurement_request.schema.json` - the request IR. Units
  mandatory; hard constraints separated from soft preferences; every
  field tagged stated/inferred/unresolved.
- `data/` - 106 records covering all 49 ALS beamlines and 57 APS
  beamlines (8 + 3 "deep" records built by verifying against instrument
  papers and science highlights; the rest directory-level). Every record
  carries provenance URLs and retrieval dates.
- `queries/matchmaker.py` - keyword+synonym ranking demo.
- `queries/ir_matcher.py` - deterministic hard-constraint matcher:
  per-constraint PASS/FAIL/UNKNOWN verdicts (missing data is UNKNOWN,
  never FAIL), explainable rejections, near-misses, clarifying questions.
- `queries/nl_to_ir.py` - natural language -> request IR via LLM
  structured extraction (uses the `claude` CLI).
- `queries/mcp_server.py` - the catalog as an MCP server (4 tools), so
  any agent can call it.

## Try it

    python3 queries/matchmaker.py "trace element distribution in plant roots"
    # ALS's XRF microprobe is not operational; the catalog answers with
    # the operational APS microprobes (GSECARS 13-ID-E, 2-ID-E) instead.

    python3 queries/ir_matcher.py queries/requests/cu_operando.json
    # structured request -> per-constraint verdicts; resolves to ALS 9.3.2,
    # verifying a 1 Torr CO2 requirement against the record's quoted 10 Torr.

    # as an MCP server (requires `pip install mcp`):
    claude mcp add capability-graph -- python3 $(pwd)/queries/mcp_server.py

## Honest limitations

- Coverage: two facilities of DOE's dozens; directory-level records have
  derived (unverified) measurables, flagged as such in output.
- Status data ages on facility-upgrade timescales; every record carries
  its retrieval date (2026-08-27).
- The keyword scorer is deliberately simple; the schema and the
  PASS/FAIL/UNKNOWN discipline are the contributions. Upgrade path:
  embedding retrieval, PaNET ontology alignment (`panet_id` is stubbed),
  LLM feasibility adjudication of the top-k.
