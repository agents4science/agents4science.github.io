# Capabilities

CAF enables agentic applications to scale from local execution to governed, autonomous scientific systems.

---

<div style="display: flex; flex-wrap: wrap; gap: 1.5rem; margin: 2rem 0;">

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3>1. <a href="local-agents.md">Local Agent Execution</a></h3>
<p>LangGraph</p>
<p>Your on-ramp to CAF. Run persistent, stateful agents on a laptop or workstation—no federation required.</p>
<p><strong>Status:</strong> <span style="color: green;">Mature</span></p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3>2. <a href="federated-agents.md">Federated Agent Execution</a></h3>
<p>LangGraph + Academy</p>
<p>Cross-institutional agent execution under federated identity and policy.</p>
<p><strong>Status:</strong> <span style="color: green;">Mature</span></p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3>3. <a href="scale-inference.md">Scale Agent Inference</a></h3>
<p>FIRST + inference orchestration</p>
<p>Fan out queries to thousands of parallel LLM instances on HPC.</p>
<p><strong>Status:</strong> <span style="color: orange;">Prototype</span></p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3>4. <a href="scale-tool-use.md">Scale and Govern Tool Use</a></h3>
<p>Governance + scheduling</p>
<p>Coordinate massive tool invocations with policy enforcement, scheduling, and auditability.</p>
<p><strong>Status:</strong> <span style="color: orange;">Prototype</span></p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3>5. <a href="coordinate-agents.md">Coordinate Many Agents</a></h3>
<p>Shared state, policy, budgets</p>
<p>Multiple agents collaborate with shared state, coordinated policies, and resource budgets.</p>
<p><strong>Status:</strong> <span style="color: blue;">Emerging</span></p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem; border-style: dashed; opacity: 0.8;">
<h3>6. <a href="autonomous-systems.md">Autonomous Scientific Systems</a></h3>
<p>Persistent, governed autonomy</p>
<p>Long-running autonomous systems that operate continuously under governance constraints.</p>
<p><strong>Status:</strong> <span style="color: gray;">Future</span></p>
</div>

</div>

---

## Maturity Levels

| Level | Meaning |
|-------|---------|
| <span style="color: green;">**Mature**</span> | Documented with working examples on this site |
| <span style="color: orange;">**Prototype**</span> | Demonstrated on DOE systems; documentation in progress |
| <span style="color: blue;">**Emerging**</span> | Active development; early adopters welcome |
| <span style="color: gray;">**Future**</span> | Planned capability; design in progress |

---

## Capability Matrix

| Stage | Capability | What you can do | CAF Components | Where it runs | Scale | Status | Examples |
|:-----:|------------|-----------------|----------------|---------------|-------|:------:|----------|
| 1 | [Local Agent Execution](local-agents.md) | Run persistent, stateful agents | LangGraph | Laptop, workstation, VM | Single agent | Mature | [AgentsLangChain](/Frameworks/AgentsLangChain/) |
| 2 | [Federated Agent Execution](federated-agents.md) | Invoke tools under federated identity | LangGraph + Academy | DOE HPC (Polaris, Aurora) | Multi-agent, multi-resource | Mature | [CharacterizeChemicals](/Frameworks/CharacterizeChemicals/) |
| 3 | [Scale Inference](scale-inference.md) | Parallel LLM queries | FIRST | Aurora, Polaris | 1000+ nodes | Prototype | Aurora 2000-node demo |
| 4 | [Scale Tool Use](scale-tool-use.md) | Governed tool execution | Academy governance | DOE HPC | 100-1000 nodes | Prototype | Coming soon |
| 5 | [Coordinate Agents](coordinate-agents.md) | Multi-agent collaboration | Shared state + policy | Distributed | Varies | Emerging | Coming soon |
| 6 | [Autonomous Systems](autonomous-systems.md) | Persistent scientific agents | Full CAF stack | Persistent | Continuous | Future | — |
