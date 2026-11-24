# The Instrument Capability Graph

*Machine-readable capability records for scientific instruments, and a matchmaker that routes measurement requests to the best operational instrument across facilities*

<img src="figure_routing.png" alt="Cross-facility routing: a natural-language measurement request resolves against capability records for 106 ALS and APS beamlines and routes around a not-operational instrument" style="max-width: 100%; margin: 1.5rem 0;">

Descriptions of what scientific facilities can do are written for people: beamline web pages, PDF specifications, proposal calls that assume a human has already chosen the instrument. An AI system planning an experiment needs an answer to a different question, in a form it can compute over: *what can this instrument measure, on what samples, under what constraints, and is it operational today?* A systematic survey ([below](#documents)) finds mature pieces — the PaNET technique ontology, NeXus/NXDL data-format contracts, the European wayforlight catalog — but no instrument- and configuration-level capability model, and no matching layer over one.

As a feasibility test, we built one: a capability catalog covering all 106 beamlines of the Advanced Light Source and the Advanced Photon Source, extracted by AI agents from public documentation at roughly one agent-run per instrument, with per-field provenance on every record. A matchmaker resolves natural-language measurement requests against the catalog.

## The demo that carries the argument

The classic instrument for mapping trace elements in plant roots is the ALS hard X-ray fluorescence microprobe, beamline 10.3.2. It is currently not operational. Because the records carry live status and span two facilities, the request routes instead to operational APS microprobes — GSECARS 13-ID-E first, whose demonstrated applications include plant roots:

```
QUERY: "trace element distribution in plant roots"
  1. [ 28.2] aps-13-id-e  13-ID-E GSECARS X-ray Microprobe
         status=operational  record_depth=deep
         techniques: micro-XRF, micro-XAFS, micro-XRD, fCMT
         matched demonstrated_applications: fluorescence, plant, roots, trace
  2. [ 26.7] aps-2-id-e  2-ID-E X-ray Fluorescence Microprobe
         status=operational  record_depth=deep
         techniques: XFM, XRF-CT, X-ray fluorescence laminography, micro-XANES
  ...
  (with --all: als-10.3.2 X-Ray Fluorescence Microprobe, status=not-operational)
```

The complex behaves as one instrument. On the eight canned demo queries, seven rank the domain-correct instrument first; the eighth is the routing case above.

## Design decisions that matter

1. **Measurables vs. techniques.** `measurables` records what a user learns ("oxidation state under reaction conditions"); `techniques` records how (APXPS). Users ask in measurable-space; catalogs are written in technique-space. Carrying both is what makes matchmaking work without requiring users to know technique names.
2. **Demonstrated applications as a first-class field**, extracted from science highlights rather than spec sheets. "Real-time micro-CT of ablators at reentry temperatures" answers feasibility questions no specification can.
3. **Provenance on every record**, per source, including conflicts recorded verbatim — one beamline's maximum pressure is stated as "above 2 mbar" in its instrument paper and "up to 10 Torr" in a facility science highlight. Capability descriptions scraped from human-oriented documentation contain such contradictions; publication by facilities as a first-class product, with provenance, is the fix.
4. **Two record depths.** Directory records are cheap (one fetch covers a facility) and give coverage; deep records cost about one agent-run each and carry sourced constraints (30 µL per well, ≤2% glycerol). Both earn their keep: directory records give breadth, deep records win rankings and answer feasibility.
5. **Status is load-bearing.** Only live status prevents routing users to dead instruments, and only a cross-facility graph can offer the alternative.

The scorer is keyword-plus-synonym overlap, deliberately simple. What is meant to carry over to a real system is the schema and the extraction recipe: the upgrade path is embeddings over measurables, technique alignment to PaNET (the field is already stubbed), and an LLM feasibility check on the top candidates.

## Measured cost

- Directory pass: one fetch covers ~50 beamlines.
- Deep pass: one agent per beamline, 6–23 tool calls, 21–37k tokens each, facts verified against instrument papers and science highlights.
- Extrapolation: all ~200 beamlines at the five DOE light sources at deep level is roughly 200 agent-runs — a day of wall-clock with modest parallelism. Neutron sources, EMSL, the Molecular Foundry, and multi-node programs such as the NSF Programmable Cloud Laboratory test bed follow the same recipe with a different `instrument_class`.

## Documents

<div style="display: flex; flex-wrap: wrap; gap: 1.5rem; margin: 2rem 0;">

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3><a href="Ian_Foster_SCAC_written_input.pdf">Written input to the DOE SCAC facilities subcommittee</a></h3>
<p><em>The argument (2026)</em></p>
<p>Why a DOE-wide machine-readable capability layer is a gap in the facilities list: AI-addressability as a design criterion, the capability layer and trust substrate as cross-cutting infrastructure, science factories as a new facility class, and the portfolio prioritized as a network rather than a list.</p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3><a href="https://github.com/agents4science/agents4science.github.io/tree/main/CapabilityGraph/prototype">The prototype</a></h3>
<p><em>Schema, 106 records, matchmaker</em></p>
<p>JSON Schema for instrument capability records and measurement requests; ALS and APS catalogs with per-field provenance; the matchmaker, a hard-constraint IR matcher with PASS/FAIL/UNKNOWN verdicts, and an MCP server that exposes the graph as agent tools. <code>python3 queries/matchmaker.py --demo</code> runs the eight canned queries.</p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3><a href="ontology_landscape.md">Ontology and catalog landscape</a></h3>
<p><em>What exists to reuse (verified survey, 2026)</em></p>
<p>PaNET, NeXus/NXDL, PIDINST, and wayforlight.eu assessed against the instrument-capability level this graph targets: each is mature and reusable, and the intersection — a per-instrument, per-configuration capability model with a matching layer — is unbuilt.</p>
</div>

</div>

## Contact

Ian Foster — [foster@uchicago.edu](mailto:foster@uchicago.edu)
