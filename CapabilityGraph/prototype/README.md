# Instrument Capability Graph — Prototype

A machine-readable catalog of scientific-instrument capabilities plus a
matchmaker that resolves "I want to measure X" to ranked instruments.
Scope: ALS (49 beamlines: 8 deep + 41 directory) and APS (57 records:
3 deep — 2-ID-E, 13-ID-E, 2-BM — + 54 directory). 106 records total.

Flagship cross-facility demo: "trace element distribution in plant roots" —
the ALS XRF microprobe (10.3.2) is not operational, so the query resolves
to the operational APS microprobes (GSECARS 13-ID-E first, whose
demonstrated applications include plant roots; 2-ID-E second).

APS extraction note: aps.anl.gov is behind Cloudflare bot protection, so
records were assembled from off-domain CAT sites (GSECARS/HPCAT/BioCAT/DCS),
GitHub/ReadTheDocs beamline docs, PMC/arXiv/OSTI papers, and search
snippets — provenance records which route supplied each field. Post-APS-U
status is sourced where possible, "unknown" otherwise (never guessed).

Selection front doors (measurement-request IR): see
`schema/measurement_request.schema.json`, `queries/nl_to_ir.py` (NL→IR via
LLM structured extraction), `queries/ir_matcher.py` (deterministic
hard-constraint filter with PASS/FAIL/UNKNOWN verdicts, near-misses, and
clarifying questions).

Built 2026-08-27 as part of the GM_Instruments project (instrument
selection / operation / design agents — this catalog is the substrate for
the "selection" capability).

## Layout

```
schema/instrument_capability.schema.json   record schema (JSON Schema 2020-12)
data/als_deep.json                         8 deep records (per-beamline extraction, full provenance)
data/als_directory.json                    41 directory records (facility table only)
data/build_directory_records.py            generator for the directory records
queries/matchmaker.py                      "measure X" -> ranked instruments demo
```

## Design decisions that matter

1. **measurables vs techniques.** `measurables` is WHAT a user learns
   ("oxidation state under reaction conditions"); `techniques` is HOW
   (APXPS). Users ask in measurable-space; catalogs are written in
   technique-space. Keeping both fields is what makes matchmaking work
   without forcing users to know technique names.
2. **demonstrated_applications as a first-class field.** Extracted from
   science highlights, not spec sheets. Demonstrated performance ("real-time
   micro-CT of ablators at reentry temperatures") answers feasibility
   questions no spec sheet can.
3. **Provenance on every record**, per-source, with notes on which fields
   came from where. Deep records also capture conflicts verbatim (e.g.
   9.3.2's max pressure: "above 2 mbar" in the 2010 instrument paper vs
   "up to 10 Torr" in a 2009 ALS science highlight (#179, reporting Tao et
   al., Science 322, 932 (2008)) — both recorded).
4. **record_depth: directory vs deep.** Directory records are cheap
   (one fetch covers a facility) but their measurables are derived from a
   technique->measurable map. Deep records cost ~1 agent each (~10 pages
   read) and carry sourced constraints (SIBYLS: 30 uL per well, <=2%
   glycerol; MERLIN: cleaved in UHV better than 8e-11 Torr). The demo shows
   both depths are useful: directory records give coverage, deep records
   win rankings and answer feasibility.
5. **status is load-bearing.** The demo query "trace element distribution
   in plant roots" has a correct answer (XRF microprobe 10.3.2) that is
   currently NOT OPERATIONAL. Only live status data prevents routing users
   to dead instruments — and only a cross-facility graph can offer the
   alternative (APS/SSRL microprobes). Several ALS records also carry
   ALS-U dark-time closure caveats: capability graphs go stale on
   facility-upgrade timescales, so retrieval dates matter.

## Run the demo

```
python3 queries/matchmaker.py --demo
python3 queries/matchmaker.py "surface oxidation state of a catalyst under reaction gas"
python3 queries/matchmaker.py --all "trace element distribution"   # include down instruments
```

Result on the 8 canned queries: 7/8 rank the domain-correct instrument
first; the 8th is the not-operational case described above.

The scorer is keyword+synonym overlap — deliberately dumb. What's meant to
carry over to a real system is the schema and the extraction recipe, not
the scorer. The upgrade path: embed measurables/applications, align
techniques to PaNET (field already stubbed: `techniques[].panet_id`), and
put an LLM feasibility check on the top-k (sample constraints vs the
user's sample).

## Extraction cost calibration (measured)

- Directory pass: 1 WebFetch for ~50 beamlines (name, techniques, energy
  range, status).
- Deep pass: 1 agent per beamline, 6-23 tool calls each, ~21-37k tokens
  each. All 8 verified their facts against instrument papers and science
  highlights, not just the beamline page.
- Extrapolation: all ~5 DOE lightsources (~200 beamlines) at deep level is
  roughly 200 agent-runs — a day of wall-clock with modest parallelism.
  Neutron sources (SNS/HFIR), EMSL, Molecular Foundry, and NIF-class
  facilities would follow the same recipe with a different
  `instrument_class`.

## Known gaps

- measurables vocabulary is uncontrolled (free text). Needs alignment to an
  ontology; see the deep-research report on PaNET/NeXus/PIDINST/wayforlight
  (`../ontology_landscape.md`) for what to reuse.
- No cross-facility records yet (single facility = no redundancy/alternatives).
- No availability/beamtime-cycle data; access mechanisms are captured as
  free text.
- Directory-derived measurables are unverified by definition (flagged in
  output).
