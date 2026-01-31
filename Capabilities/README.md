# Capabilities

*Deployment patterns from local execution to autonomous systems*

CAF enables agentic applications to scale from local execution to governed, autonomous scientific systems.

---

<div style="display: flex; flex-wrap: wrap; gap: 1.5rem; margin: 2rem 0;">

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3>1. <a href="local-agents/">Local Agent Execution</a></h3>
<p>LangGraph, Academy</p>
<p>Your on-ramp to CAF. Run persistent, stateful agents on a laptop or workstation—no federation required.</p>
<p><strong>Status:</strong> <span style="color: green;">Mature</span></p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3>2. <a href="federated-agents/">Federated Agent Execution</a></h3>
<p>LangGraph + Academy</p>
<p>Cross-institutional agent execution under federated identity and policy.</p>
<p><strong>Status:</strong> <span style="color: green;">Mature</span></p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3>3. <a href="scale-inference/">Massively Parallel Agent Inference</a></h3>
<p>LangGraph, Aegis</p>
<p>Fan out thousands of LLM requests in parallel on HPC.</p>
<p><strong>Status:</strong> <span style="color: orange;">Prototype</span></p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3>4. <a href="governed-tool-use/">Governed Tool Use at Scale</a></h3>
<p>Academy governance</p>
<p>Invoke expensive, stateful, or dangerous tools under proactive policy enforcement.</p>
<p><strong>Status:</strong> <span style="color: orange;">WIP</span></p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3>5. <a href="multi-agent-coordination/">Multi-Agent Coordination</a></h3>
<p>Shared state + policy + budgets</p>
<p>Many agents under shared governance—within one institution or across many.</p>
<p><strong>Status:</strong> <span style="color: blue;">Emerging</span></p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem;">
<h3>6. <a href="long-lived-agents/">Long-Lived Autonomous Agents</a></h3>
<p>Lifecycle management</p>
<p>Agents that persist for days to months, maintaining state, memory, and goals.</p>
<p><strong>Status:</strong> <span style="color: blue;">Emerging</span></p>
</div>

<div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem; border-style: dashed; opacity: 0.8;">
<h3>7. <a href="agent-workflows/">Agent-Mediated Scientific Workflows</a></h3>
<p>Dynamic workflow construction</p>
<p>Agents dynamically construct, adapt, and execute scientific workflows.</p>
<p><strong>Status:</strong> <span style="color: blue;">Early</span></p>
</div>

</div>

---

## Maturity Levels

| Level | Meaning |
|-------|---------|
| <span style="color: green;">**Mature**</span> | Documented with working examples on this site |
| <span style="color: orange;">**Prototype**</span> | Demonstrated on DOE systems; documentation in progress |
| <span style="color: orange;">**WIP**</span> | Work in progress |
| <span style="color: blue;">**Emerging**</span> | Active development; early adopters welcome |
| <span style="color: blue;">**Early**</span> | Early stage; design and prototyping |

---

## Capability Matrix

| Stage | Capability | What you can do | CAF Components | Where it runs | Scale | Status |
|:-----:|------------|-----------------|----------------|---------------|-------|:------:|
| 1 | [Local Agent Execution](local-agents/) | Run persistent, stateful agents | LangGraph | Laptop, workstation, VM | Single agent | Mature |
| 2 | [Federated Agent Execution](federated-agents/) | Invoke tools under federated identity | LangGraph + Academy | DOE HPC | Multi-resource | Mature |
| 3 | [Parallel Agent Inference](scale-inference/) | Fan out thousands of LLM requests | LangGraph + FIRST | HPC accelerators | O(10³–10⁴) streams | Prototype |
| 4 | [Governed Tool Use](governed-tool-use/) | Invoke tools under policy enforcement | Academy governance | DOE HPC | O(10²–10³) tools | WIP |
| 5 | [Multi-Agent Coordination](multi-agent-coordination/) | Coordinate agents under shared governance | Shared state + policy | Distributed | O(10²–10³) agents | Emerging |
| 6 | [Long-Lived Agents](long-lived-agents/) | Persistent agents with memory and goals | Lifecycle management | Any | Days–months | Emerging |
| 7 | [Agent Workflows](agent-workflows/) | Dynamic workflow construction | Workflow integration | DOE infrastructure | Varies | Early |
