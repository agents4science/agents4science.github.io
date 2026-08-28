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
- `schema/measurement_request.schema.json` - the measurement-request intermediate representation (IR). Units
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

The ALS XRF microprobe (10.3.2) is not operational, so the catalog
answers cross-facility with the operational APS microprobes:

```

QUERY: "trace element distribution in plant roots"
  1. [ 28.2] aps-13-id-e  13-ID-E GSECARS X-ray Microprobe
         status=operational  record_depth=deep
         techniques: micro-XRF, micro-XAFS, micro-XRD, fCMT
         matched measurables: composition, distribution, element, elemental, fluorescence, trace
         matched demonstrated_applications: fluorescence, plant, roots, trace
  2. [ 26.7] aps-2-id-e  2-ID-E X-ray Fluorescence Microprobe
         status=operational  record_depth=deep
         techniques: XFM, XRF-CT, X-ray fluorescence laminography, micro-XANES
         matched measurables: composition, distribution, element, elemental, trace
  ...
```

    python3 queries/ir_matcher.py queries/requests/cu_operando.json

A structured request (operando Cu oxidation state, 1 Torr CO2) gets
per-constraint verdicts; `+` = verified pass, `?` = unknown (missing
data is never treated as failure):

```
REQUEST: We have a 10 mm copper catalyst pellet and want to follow the oxidation state of the Cu surface while it is exposed to about 1 Torr of CO2 at room temperature. The measurement must be element-specific. The sample must not be powdered. Non-destructive strongly preferred; we don't need spatial mapping.

=== CANDIDATES (3) ===

1. [ 24.1] als-9.3.2  Beamline 9.3.2 — Ambient-Pressure Soft X-Ray Photoelectron Spectroscopy  (depth=deep)
     + status                     operational
     + sample state               accepts 'solid'
     ? vacuum tolerance           vacuum tolerance not constrained
     + in-situ gas environment    stated gas-pressure capability 10 torr >= requested 1 Torr
     + element specificity        offers an element-specific technique
     ? edge coverage              no absorption edge of ['Cu'] in 200-900 eV, but photoemission probes core-level binding energies instead - needs adjudication against core levels
     ? spatial resolution         no spatial-resolution requirement

  ...
```

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
