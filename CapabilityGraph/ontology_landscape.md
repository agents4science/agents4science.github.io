# Ontology & Catalog Landscape for an Instrument Capability Graph

Deep-research report, 2026-08-27. Method: 5 parallel search angles, 20 primary
sources fetched, 99 claims extracted, top 25 adversarially verified with 3
independent votes each (22 confirmed, 3 refuted). Confidence labels reflect
that verification. Companion prototype: `capability_graph/`.

## Bottom line

Three mature, reusable assets exist — **PaNET** (technique vocabulary),
**NeXus/NXDL** (per-technique configuration-parameter contracts), and
**wayforlight.eu** (the operational model for a cross-facility beamline
catalog). **None of them operates at the instrument-capability level a DOE
graph targets.** The intersection — a semantically formal, per-instrument,
per-configuration capability model with a measurement-need matching layer —
is genuinely unbuilt. That is precisely the layer the `capability_graph/`
prototype models.

## What exists (verified, high confidence)

### PaNET — Photon and Neutron Experimental Techniques ontology
- OWL ontology, ~374–400 technique classes (126 core), poly-hierarchical,
  global persistent IRIs (`http://purl.org/pan-science/PaNET/PaNETxxxxxxx`).
  Created 2021 under ExPaNDS WP3 with PaNOSC. **Technique-level only: models
  no instruments or configurations.**
- Fully machine-readable: OWL generated from CSV via ROBOT; published in
  JSON-LD, RDF/XML, N-Triples, Turtle; SPARQL + REST via NCBO BioPortal.
  Repo moved to `github.com/pan-ontologies/PaNET`; commits through July 2026,
  weekly maintenance meetings.
- In production use: PaNOSC federated search consumes PaNET dataset tags;
  Diamond plans a PaNET term selector in DataGateway; DESY started a
  web-based technique selector.
- Known gap, fix in flight: hierarchy was built by manual subclass assertion
  (a query for "x-ray scattering technique" subclasses fails to return
  "x-ray diffraction technique"). The **PaNET 2.0** redesign (J. Synchrotron
  Rad. 32, 1361–1369, 2025; DOI 10.1107/S1600577525005272) adds four object
  properties — definedByProbe, definedByProcess, definedByDependence,
  definedByPurpose — plus 128 concept classes so an OWL reasoner infers
  subsumption (117 new subclass relations inferred, 102 verified correct).
  Proposed in the 2025 paper; not confirmed merged into the release.
- Design intent is **dataset tagging/search, not capability matching** —
  reuse it as the technique axis, not as the capability model.
- Sources: journals.iucr.org/s/issues/2025/05/00/ok5142/,
  github.com/ExPaNDS-eu/ExPaNDS-experimental-techniques-ontology,
  pan-ontologies.github.io/PaNET/.

### NeXus / NXDL
- NXDL application definitions are XML-Schema-validated, machine-readable
  data contracts at technique/discipline granularity (e.g. NXsas covers all
  monochromatic area-detector SAS: SANS, SAXS, WSAS, GISAS, both probes).
  Verifier reproduced `xmllint --schema nxdl.xsd NXsas.nxdl.xml` → validates.
  Actively maintained by NIAC (v2026.01).
- Definitions carry enumerated controlled values (probe ∈ {neutron, x-ray})
  and a ready configuration-parameter vocabulary (NXsource probe/type,
  NXmonochromator wavelength(+spread, optional), NXdetector distance, pixel
  sizes). **A capability graph should parse NXDL programmatically for
  per-technique configuration schemas rather than re-encoding them.**
- Not a facility/instrument capability listing — describes data files and
  components.
- Sources: manual.nexusformat.org/nxdl.html,
  github.com/nexusformat/definitions.

### wayforlight.eu
- The closest existing analog: live (2026) catalogue of ~300 European
  synchrotron/FEL beamlines; faceted search by discipline, technique, energy
  range, sample type; **standardized datasheets** enabling cross-facility
  comparison; APIs let beamline scientists enter data once for republication
  on facility sites (enter-once/republish).
- Caveats: APIs are internal to participating facilities (not publicly
  documented); individual datasheets can be stale; EC trans-national-access
  funding ended Oct 2021 though the catalogue remains maintained. Structured
  web data, **not a formal ontology**.
- Sources: tandfonline.com/doi/full/10.1080/08940886.2019.1582291 (SRN 2019),
  wayforlight.eu (live check 2026-08-27).

### PaNmapping (ExPaNDS/PaNOSC alignment ontology)
- v0.1 (2021, HZB custodian): SKOS/OWL crosswalks among PaNET, NeXus
  ontology, ICAT CSMD, and generic standards (DCAT v2, DublinCore, FOAF,
  PROV-O, schema.org). The precedent for the crosswalk layer a DOE graph
  needs. Current maintenance status unverified.

## Reuse vs build (synthesis)

**Reuse:**
1. PaNET IRIs as the canonical technique vocabulary — ideally the PaNET 2.0
   probe/process/dependence/purpose axes for compositional reasoning
   (the prototype schema already stubs `techniques[].panet_id`).
2. NXDL application definitions, parsed programmatically, as per-technique
   configuration/parameter vocabularies.
3. PaNmapping-style SKOS/OWL crosswalks to DCAT/schema.org for web/EOSC
   interoperability.
4. wayforlight's operational model: standardized datasheets +
   enter-once/API-republish (adoption incentive for facilities).

**Build new (verified to exist nowhere):**
1. The instrument- and configuration-level capability model:
   facility → instrument → configuration → achievable parameter ranges,
   sample constraints, demonstrated applications.
2. Technique-to-instrument capability linking.
3. The measurement-need → instrument matching/recommendation layer.

## Explicitly unmapped (do not treat as "doesn't exist")

No claims survived verification for: DOE facility machine-readable capability
listings (ALS, APS, NSLS-II, LCLS, SSRL, SNS/HFIR, EMSL, Molecular Foundry,
NERSC ScienceSearch), PIDINST, OSTI instrument records, Helmholtz/HMC
instrument metadata, schema.org instrument vocabularies, ESRF/ICAT specifics,
and AI beamline recommenders. These need a targeted second research pass.
(Note: our own prototype confirms ALS publishes a human-readable directory
table but no API/schema was found.)

## Refuted during verification (avoid these framings)

- "PaNET defines 1,300+ classes" — refuted; correct figure ~374–400.
- A specific NeXusOntology characterization (auto-generated OWL rendering
  claim as stated) — refuted 0-3.
- "ExPaNDS grant 823852, active 2019–2023, ten facilities" — refuted as
  stated (details wrong).

## Open questions for the next pass

1. Do any DOE facilities expose machine-readable capability listings/APIs
   comparable to wayforlight, and in what schema?
2. Current status/adoption of PIDINST, OSTI instrument records, HMC — could
   PIDINST be the instrument-identity backbone of the graph?
3. Is PaNET 2.0's reasoning structure merged into the released ontology?
   Is PaNmapping maintained beyond v0.1?
4. Do operational technique/beamline recommenders exist (DESY selector,
   Diamond DataGateway, ScienceSearch), and what representations do they
   consume?
