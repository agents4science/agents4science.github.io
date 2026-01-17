# Agent Frameworks


An **agent** is a persistent, stateful process that acts on behalf of a user or system.  An agent may:
<div style="display: flex; gap: 1rem; align-items: flex-start;">
  <div style="flex: 1;">
    <ul>
      <li>Observe inputs or events</li>
      <li>Plan (decide on) actions using a policy (rules or LLM)</li>
      <li>Act: Execute tools or interact with other agents</li>
      <li>Learn: Update state to adapt over time</li>
    </ul>
  </div>
  <img src="Assets/agent.png"
       alt="Agent"
       style="width: 180px; height: auto;">
</div>
We can think of an agent as a *scientific assistant that can reason, act, and coordinate on our behalf*.

An **agent framework** provides abstractions and runtime support that simplify the development and use of agents.
We focus here on three agent frameworks that are well-suited for scientific applications:

* **LangGraph** offers structured, reproducible workflows for LLM-driven reasoning and tool execution. Good for managing interactions with LLMs and for implementing structured or auditable reasoning pipelines.

* **Academy** provides persistent, secure, and scalable execution across HPC systems, instruments, and data services. Good for agents that must run continuously or securely on HPC systems, laboratory robots, data platforms, or other parts of federated DOE infrastructure.

* **Microsoft Agent Framework** (MAF) supports flexible multi-agent coordination patterns and conversational planning. Good for multi-agent coordination, committee-based reasoning, or adaptive planning strategies involving interaction among several agents.

See [these slides](https://docs.google.com/presentation/d/1Djvi5_PqvZl1v1xO2nWJf3k7P-35XGcH) for a brief review of these systems (and one more, **Microsoft Agent Framework**).




## A simple LangGraph Example

[AgentsLangChain](AgentsLangChain) implements a simple multi-agent example using LangChain.

## A Simple Academy Example

TBD

## Scalability

A frequent requirement, at least in our agentic applications, is to scale the number of ...
