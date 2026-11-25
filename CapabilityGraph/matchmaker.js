// Client-side port of prototype/queries/matchmaker.py — same synonyms,
// field weights, and scoring, so the browser reproduces the Python demo.

const SYNONYMS = {
  "3d": ["tomography", "tomogram", "microstructure", "internal", "morphology"],
  "microstructure": ["morphology", "tomography", "porosity", "3d"],
  "porosity": ["pore", "microstructure", "tomography"],
  "crack": ["microstructure", "morphology", "in-situ", "evolution"],
  "band": ["electronic", "fermi", "arpes", "dispersion"],
  "electronic": ["band", "fermi", "arpes", "excitations", "states"],
  "magnetic": ["magnetization", "spin", "domain"],
  "spin": ["magnetic", "domain"],
  "protein": ["macromolecular", "macromolecule", "solution", "crystallography"],
  "structure": ["structural", "crystal", "lattice"],
  "crystal": ["crystallography", "diffraction", "lattice", "structure"],
  "pressure": ["gpa", "anvil", "extreme", "compression"],
  "catalyst": ["catalysis", "surface", "reaction", "operando"],
  "surface": ["interface", "interfacial", "adsorbate"],
  "oxidation": ["redox", "valence", "chemical state", "speciation"],
  "redox": ["oxidation", "valence"],
  "chemical": ["speciation", "composition", "bonding"],
  "battery": ["electrode", "cathode", "anode", "electrochemistry", "lithium"],
  "electrode": ["battery", "electrochemistry"],
  "operando": ["in-situ", "in situ", "reaction conditions", "real-time"],
  "in-situ": ["operando", "real-time", "evolution"],
  "element": ["elemental", "composition", "fluorescence"],
  "trace": ["elemental", "fluorescence", "microprobe"],
  "map": ["maps", "mapping", "imaging", "microscopy", "distribution"],
  "image": ["imaging", "microscopy", "map"],
  "nanoscale": ["nanometer", "nano", "nm", "ptychography"],
  "vibrational": ["infrared", "functional-group", "molecular"],
  "biofilm": ["biological", "bacteria", "living"],
  "living": ["biological", "real-time", "hydrated"],
  "phase": ["transition", "crystalline", "diffraction"],
  "strain": ["stress", "lattice", "elastic", "microdiffraction"],
  "solution": ["saxs", "scattering", "conformational"],
  "shape": ["envelope", "conformational", "saxs", "morphology"],
  "flexibility": ["conformational", "ensembles", "solution"],
  "gas": ["ambient-pressure", "reaction conditions"],
  "molecular": ["molar", "molecular weight", "sec", "gpc", "mals", "functional-group", "vibrational", "infrared"],
  "weight": ["molar mass", "mass", "dispersity", "sec"],
  "glass": ["transition", "thermal", "dsc", "calorimetry"],
  "transition": ["glass", "thermal", "melting", "crystallization"],
  "thermal": ["glass transition", "stability", "decomposition", "dsc", "tga", "heat"],
  "viscosity": ["rheology", "flow", "rod-pull", "viscometry"],
  "rheology": ["viscosity", "flow", "jamming", "modulus"],
  "zeta": ["electrophoretic", "charge", "dls", "colloid"],
  "modulus": ["hardness", "mechanical", "nanoindentation", "stiffness"],
  "hardness": ["modulus", "mechanical", "nanoindentation", "scratch"],
  "gloss": ["sheen", "reflectivity", "coating"],
  "synthesize": ["synthesis", "polymerization", "formulation", "library"],
  "polymer": ["copolymer", "latex", "film", "coating", "macromolecule"],
  "film": ["thin film", "coating", "casting"],
  "particle": ["colloid", "dispersion", "size", "zeta"],
  "tissue": ["biological", "chemical maps", "micron"],
};

const STOP = new Set("i want to measure of a the in on at an and for with under during my our its is are how what best".split(" "));

const FIELD_WEIGHTS = [
  ["measurables", 3.0],
  ["demonstrated_applications", 2.0],
  ["techniques", 2.0],
  ["sample_environments", 1.5],
  ["name", 1.0],
  ["aliases", 1.0],
  ["facility", 3.0],   // identity: being AT a facility outranks being named like one
  ["institution", 1.0],
  ["program", 1.0],
  ["location", 1.0],
];

function tokens(text) {
  const out = [];
  for (const m of text.toLowerCase().matchAll(/[a-z0-9][a-z0-9\-]+/g)) {
    if (!STOP.has(m[0])) out.push(m[0]);
  }
  return out;
}

function expand(queryTokens) {
  const expanded = new Map();
  for (const t of queryTokens) if (!expanded.has(t)) expanded.set(t, 1.0);
  for (const t of queryTokens) {
    const base = t.replace(/s+$/, "");
    for (const [key, syns] of Object.entries(SYNONYMS)) {
      if (t === key || base === key) {
        for (const s of syns) if (!expanded.has(s)) expanded.set(s, 0.5);
      }
    }
  }
  return expanded;
}

function fieldText(record, field) {
  const v = record[field] ?? "";
  if (field === "location") {
    return [v.building, v.room, v.address].filter(Boolean).join(" ");
  }
  if (field === "techniques") {
    const parts = [];
    for (const t of v) {
      parts.push(t.name || "", t.acronym || "", ...(t.variants || []));
    }
    return parts.join(" ");
  }
  if (Array.isArray(v)) return v.map(String).join(" ");
  return String(v);
}

function scoreRecord(record, expanded) {
  let total = 0.0;
  const matched = {};
  for (const [field, weight] of FIELD_WEIGHTS) {
    const text = fieldText(record, field).toLowerCase();
    for (const [term, tw] of expanded) {
      if (text.includes(term)) {
        total += weight * tw;
        (matched[field] ??= new Set()).add(term);
      }
    }
  }
  if (record.record_depth === "deep") total *= 1.15;
  return [total, matched];
}

const EXCLUDED_STATUS = new Set(["closed", "not-operational"]);

// Returns {results, excluded}: excluded holds would-have-matched records
// filtered out by status, so the UI can surface the routing story.
function search(records, query, includeAll = false) {
  const expanded = expand(tokens(query));
  const results = [], excluded = [];
  for (const r of records) {
    const [s, matched] = scoreRecord(r, expanded);
    if (s <= 0) continue;
    if (!includeAll && EXCLUDED_STATUS.has(r.status)) {
      excluded.push({ score: s, record: r, matched });
    } else {
      results.push({ score: s, record: r, matched });
    }
  }
  results.sort((a, b) => b.score - a.score);
  excluded.sort((a, b) => b.score - a.score);
  return { results, excluded };
}

const DEMO_QUERIES = [
  "3d microstructure of a battery electrode during in-situ cycling",
  "electronic band structure and fermi surface of a 2d quantum material",
  "protein solution shape and conformational flexibility",
  "surface oxidation state of a catalyst under reaction gas conditions",
  "crystal structure phase transition at high pressure",
  "chemical composition map of living bacterial biofilm",
  "magnetic domain structure of a thin film at the nanoscale",
  "trace element distribution in plant roots",
];

if (typeof module !== "undefined") {
  module.exports = { tokens, expand, scoreRecord, search, DEMO_QUERIES };
}
