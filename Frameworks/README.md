# Agent Frameworks

An agent framework provides abstractions and runtime support for building persistent computational entities that perceive state, reason over goals, take actions, and coordinate with other agents under resource and policy constraints.
In the scientific context, we also want a framework to enable interactions with scientific resources, such as computers and instruments.

We focus here on two agent frameworks here: **LangGraph** and **Academy**, due to their excellent support for 
managing interactions with LLMs (LangGraph) and scientific resources (Academy).
See [these slides](https://docs.google.com/presentation/d/1Djvi5_PqvZl1v1xO2nWJf3k7P-35XGcH) for a brief review of these systems (and one more, **Microsoft Agent Framework**).


CAF also works with AmSC & other ModCon teams to provide abstractions, infrastructure services, & design guidance that allow agents (whether implemented with agent frameworks or custom code) to run securely, scalably, & safely across DOE HPC, instruments, etc., as well as cloud, while ensuring portability, interoperability, & policy compliance.

## A simple LangGraph Example

[AgentsLangChain](AgentsLangChain) implements a simple multi-agent example using LangChain.

## A Simple Academy Example

TBD

## Scalability

A frequent requirement, at least in our agentic applications, is to scale the number of ...
