# Capabilities

The Common Agentic Framework (CAF) supports a range of deployment patterns for scientific agentic applications—from simple local prototypes to massively parallel inference on DOE supercomputers.

---

## Deployment Patterns

<div style="display: flex; flex-wrap: wrap; gap: 1.5rem; margin: 2rem 0;">

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3><a href="local-agents.md">Local Agentic Applications</a></h3>
<p>Run agents on a laptop or workstation with LLM access via API or local model.</p>
<p><strong>Status:</strong> <span style="color: green;">Mature</span></p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3><a href="hpc-tools.md">Agents with HPC Tools</a></h3>
<p>Agentic applications that invoke simulation codes, data services, or other tools on DOE HPC systems.</p>
<p><strong>Status:</strong> <span style="color: green;">Mature</span></p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3><a href="parallel-inference.md">Massively Parallel Inference</a></h3>
<p>Fan out queries to thousands of parallel LLM instances on HPC.</p>
<p><strong>Status:</strong> <span style="color: orange;">Prototype</span></p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3><a href="simulation-steering.md">Simulation Steering</a></h3>
<p>Agents monitor running simulations and adjust parameters in real-time.</p>
<p><strong>Status:</strong> <span style="color: orange;">Prototype</span></p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3><a href="instrument-integration.md">Instrument Integration</a></h3>
<p>Agents orchestrate data collection from beamlines, microscopes, and lab instruments.</p>
<p><strong>Status:</strong> <span style="color: blue;">Emerging</span></p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3><a href="human-in-loop.md">Human-in-the-Loop</a></h3>
<p>Agents propose actions and wait for scientist approval at key decision points.</p>
<p><strong>Status:</strong> <span style="color: green;">Mature</span></p>
</div>

</div>

---

## Maturity Levels

| Level | Meaning |
|-------|---------|
| <span style="color: green;">**Mature**</span> | Documented with working examples on this site |
| <span style="color: orange;">**Prototype**</span> | Demonstrated on DOE systems; documentation in progress |
| <span style="color: blue;">**Emerging**</span> | Active development; early adopters welcome |

---

## Quick Reference

| Capability | Framework | Scale | Example |
|------------|-----------|-------|---------|
| Local agents | LangGraph | 1 node | [AgentsLangChain](/Frameworks/AgentsLangChain/) |
| HPC tools | LangGraph + Academy | 1-100 nodes | [CharacterizeChemicals](/Frameworks/CharacterizeChemicals/) |
| Parallel inference | Academy | 1000+ nodes | Aurora 2000-node demo |
| Simulation steering | Academy | Varies | Coming soon |
| Instrument integration | Academy + Facility APIs | 1 node + instruments | Coming soon |
| Human-in-the-loop | LangGraph interrupts | 1 node | [AgentsExample](/Frameworks/AgentsExample/) |
