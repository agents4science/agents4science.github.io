# Agent Frameworks

<div style="display: flex; gap: 2rem; align-items: flex-start;">

  <!-- LEFT: all text -->
  <div style="flex: 1;">
    <p>
      An agent is a persistent, stateful process that acts on behalf of a user
      or system. An agent may:
    </p>

    <ul>
      <li>Observe inputs or events</li>
      <li>Plan (decide on) actions using a policy (rules or LLM)</li>
      <li>Act: Execute tools or interact with other agents</li>
      <li>Learn: Update state to adapt over time</li>
    </ul>

    <p>
      We can think of an agent as a
      <em>scientific assistant that can reason, act, and coordinate on our behalf</em>.
    </p>
  </div>

  <!-- RIGHT: image -->
  <img src="Assets/agent.png"
       alt="Agent loop diagram"
       style="width: 180px; height: auto; flex-shrink: 0;">

</div>

An **agent framework** provides abstractions and runtime support that simplify the development and use of agents.
We focus here on three agent frameworks that are well-suited for scientific applications:

* [**LangGraph**](https://www.langchain.com/langgraph) offers structured, reproducible workflows for LLM-driven reasoning and tool execution. Good for managing interactions with LLMs and for implementing structured or auditable reasoning pipelines.

* [**Academy**](https://academy-agents.org) provides persistent, secure, and scalable execution across HPC systems, instruments, and data services. Good for agents that must run continuously or securely on HPC systems, laboratory robots, data platforms, or other parts of federated DOE infrastructure.

* [**Microsoft Agent Framework**](https://github.com/microsoft/agent-framework) (MAF) supports flexible multi-agent coordination patterns and conversational planning. Good for multi-agent coordination, committee-based reasoning, or adaptive planning strategies involving interaction among several agents.

See [these slides](https://docs.google.com/presentation/d/1Djvi5_PqvZl1v1xO2nWJf3k7P-35XGcH) for a brief review of these systems.



## Examples

- [**AgentsLangChain**](AgentsLangChain/) — Multi-agent pipeline using LangChain
- [**AgentsAcademy**](AgentsAcademy/) — Multi-agent pipeline using Academy
- [**AgentsExample**](AgentsExample/) — Dashboard demo with configurable goals
- [**CharacterizeChemicals**](CharacterizeChemicals/) — LLM-planned molecular property agent using Academy
