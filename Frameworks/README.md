# Agent Frameworks

An agent is a persistent, stateful process that acts on behalf of a user or system.  An agent may:

* Observe inputs or events
* Plan (decide on) actions using a policy (rules or LLM)
* Act: Execute tools or call other agents
* Learn: Update state to adapt over time

<div style="display: flex; gap: 1rem; align-items: flex-start;">
  <div style="flex: 1;">
    <ul>
      <li>Observe inputs or events</li>
      <li>Plan (decide on) actions using a policy (rules or LLM)
      <li>Act: Execute tools or call other agents
      <li>Learn: Update state to adapt over time
    </ul>
  </div>
  <img src="Assets/agent.png"
       alt="Agent"
       style="width: 100px; height: auto;">
</div>

We can think of an agent as a scientific assistant that can reason, act, and coordinate on our behalf


An agent framework provides abstractions and runtime support for building persistent computational entities that perceive state, reason over goals, take actions, and coordinate with other agents under resource and policy constraints.
In the scientific context, we also want a framework to enable interactions with scientific resources, such as computers and instruments.

We focus here on two agent frameworks here: **LangGraph** and **Academy**, due to their excellent support for 
managing interactions with LLMs (LangGraph) and scientific resources (Academy).

* LangGraph offers structured, reproducible workflows for LLM-driven reasoning and tool execution. Good for managing interactions with LLMs and for implementing structured or auditable reasoning pipelines.

* Academy provides persistent, secure, and scalable execution across HPC systems, instruments, and data services. Good for agents that must run continuously or securely on HPC systems, laboratory robots, data platforms, or other parts of federated DOE infrastructure.

See [these slides](https://docs.google.com/presentation/d/1Djvi5_PqvZl1v1xO2nWJf3k7P-35XGcH) for a brief review of these systems (and one more, **Microsoft Agent Framework**).
* Microsoft Agent Framework (MAF) supports flexible multi-agent coordination patterns and conversational planning. Good for multi-agent coordination, committee-based reasoning, or adaptive planning strategies involving interaction among several agents.




## A simple LangGraph Example

[AgentsLangChain](AgentsLangChain) implements a simple multi-agent example using LangChain.

## A Simple Academy Example

TBD

## Scalability

A frequent requirement, at least in our agentic applications, is to scale the number of ...
