# Agentic Scientific Discovery Platforms

The following **curriculum** outlines topics to be covered and readings, and provides the slides presented in class (minus purely administrative material).

---

## Week 1

### Mon Sept 29 — Lecture 1: What is an agent?
Introduces AI agents and and the  sense-plan-act-learn loop. Motivates scientific Discovery Platforms (SDPs): AI-native systems that connect reasoning models with scientific resources. 

*Slides:* [Lecture 1 slides](Assets/Lecture_1_web.pdf).

*Readings:*
- *[Exploring Large Language Model based Intelligent Agents: Definitions, Methods, and Prospects](https://arxiv.org/abs/2401.03428)*, Cheng et al. (Arxiv, 2024).
- *[Artificial intelligence and illusions of understanding in scientific research](https://www.nature.com/articles/s41586-024-07146-0)*, Messeri & Crockett (Nature, 2023).
- *[The Shift from Models to Compound AI Systems – The Berkeley Artificial Intelligence Research Blog](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/).*

### Wed Oct 1 — Lecture 2: Frontiers of Language Models
Surveys frontier reasoning models: general-purpose LLMs (GPT, Claude), domain-specific foundation models (materials, bio, weather), and hybrids. Covers techniques for eliciting better reasoning: prompting, chain-of-thought, retrieval-augmented generation (RAG), fine-tuning, and tool-augmented reasoning.

*Slides:* [Lecture 2 slides](Assets/Lecture_2_web.pdf).

*Readings:* 
- *[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)*.
- *[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948).*
- *[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629).*

*Assignment A1:* Implement a ReACT style agent.  

---

## Week 2

### Mon Oct 6 — Lecture 3: Systems for Agents
Discusses architectures and frameworks for building multi-agent systems, with emphasis on inter-agent communication, orchestration, and lifecycle management. 

*Slides:* [Lecture 3 slides](Assets/Lecture_3_web.pdf).

*Readings:*
- *[AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)*.
- *[LangGraph](https://langchain-ai.github.io/langgraph/?ajs_aid=70d48da3-5563-4143-ba79-957377dfcf92)*.
- *[AIOS: LLM Agent Operating System](https://arxiv.org/pdf/2403.16971)*.
 
### Wed Oct 8 — Lecture 4: Retrieval Augmented Generation (RAG) and Vector Databases
Covers how to augment reasoning models with external knowledge bases, vector search, and hybrid retrieval methods. 

*Slides:* [Lecture 4 slides](Assets/Lecture_4_web.pdf).

*Readings:*
- *[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)*.
- *[The FAISS library](https://arxiv.org/abs/2401.08281)*.

*Assignment A2:* Hybrid retrieval.  

---

## Week 3

### Mon Oct 13 — Lecture 5: Tool Calling
Introduces methods for invoking external tools from reasoning models. Focus on model context protocol (MCP), schema design, and execution management.

*Slides:* [Lecture 5 slides](Assets/Lecture_5_web.pdf).

*Readings:*  
- *[Introduction - Model Context Protocol](https://modelcontextprotocol.io/).*

### Wed Oct 15 — Lecture 6: HPC Systems and Self Driving Labs
How SDPs connect to HPC workflows and experimental labs. Covers distributed coordination, robotics, and federated agents. 

*Slides:* [Lecture 6 slides](Assets/Lecture_6_web.pdf).

*Readings:*  
- *[Self-Driving Laboratories for Chemistry and Materials Science](https://pubs.acs.org/doi/10.1021/acs.chemrev.4c00055)*, *Chemical Reviews*.
- *[Empowering Scientific Workflows with Federated Agents](https://arxiv.org/abs/2505.05428)*.

*Assignment A3:* Implement Distributed Battleship (and/or Implement MCP toolbox).

---

## Week 4

### Mon Oct 20 — Lecture 7: Human–AI Workflows
Explores how scientists and agents collaborate: trust boundaries, interaction design, and debugging.  

*Readings:*  
- *[Guidelines for Human-AI Interaction](https://dl.acm.org/doi/10.1145/3290605.3300233)*, Amershi et al. (*CHI*, 2019).  
- *[Interactive Debugging and Steering of Multi-Agent AI Systems](https://dl.acm.org/doi/10.1145/3706598.3713581)* (*CHI*, 2025).  

### Wed Oct 22 — Lecture 8: Benchmarking and Evaluation
Frameworks for assessing agents and SDPs: robustness, validity, and relevance.  

*Readings:*  
- *[MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://arxiv.org/abs/2410.07095).*  
- *[AI Agents That Matter](https://arxiv.org/abs/2407.01502).*  
- *[EAIRA: Establishing a Methodology for Evaluating AI Models as Scientific Research Assistants](https://arxiv.org/abs/2502.20309).*  

---

## Week 5

### Mon Oct 27 — Lecture 9: Failures and Safety
Examines why multi-agent systems fail and methods for safety and guardrails.  

*Readings:*  
- *[Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2408.00154).*  
- *[AGrail: A Lifelong Agent Guardrail with Effective and Adaptive Safety Detection](https://arxiv.org/abs/2502.11448).*  
- *[Improve accuracy by adding Automated Reasoning checks in Amazon Bedrock Guardrails](https://aws.amazon.com/bedrock/).*

*Assignment A4:* Implement evaluation harness.  

### Wed Oct 29 — Lecture 10: Case Studies
Case studies of SDPs in biology and materials.  

*Readings:*  
- *[The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies](https://www.nature.com/articles/s41586-023-06456-y)*, *Nature*.  
- *[The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search](https://arxiv.org/abs/2504.08066).*  

---

## Week 6

### Mon Nov 3 — Lecture 11: Novelty and Plagiarism
Explores originality, credit, and the risks of plagiarism in AI-generated science.  

*Readings:*  
- *[Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers](https://arxiv.org/abs/2304.12519).*  
- *[All That Glitters is Not Novel: Plagiarism in AI Generated Research](https://arxiv.org/pdf/2502.16487).*  

*Assignment A5:* Capstone project planning (novel contributions).  

### Wed Nov 5 — Lecture 12: Building Agents and Workflows
Pipelines, workflow composition, and self-improving systems.  

*Readings:*  
- *[AFlow: Automating Agentic Workflow Generation](https://arxiv.org/abs/2410.10762).*  
- *[DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714).*  

*Assignment A6:* Generating HPC workflows.  

---

## Week 7

### Mon Nov 10 — Lecture 13: Finetuning
Covers approaches to adapt agents with reinforcement learning and real-world training.  

*Readings:*  
- *[OpenPipe/ART: Agent Reinforcement Trainer](https://github.com/OpenPipe/ART).*  
- *[Agent Lightning: Train ANY AI Agents with Reinforcement Learning](https://arxiv.org/abs/2508.03680).*  

### Wed Nov 12 — Lecture 14: Responsible SDPs
Discusses ethical and policy dimensions: dual-use concerns, bias, carbon footprint, open science vs IP.  

*Suggested Readings:*  
- *[Capabilities and risks from frontier AI](https://www.gov.uk/government/publications/frontier-ai-capabilities-and-risks-discussion-paper)* (AI Safety Summit, 2024).  
- UNESCO AI Ethics framework.  
- Carbon-aware computing literature.  

---

## Week 8

### Mon Nov 17 — Lecture 15: Scaling SDPs [SC week]
Strategies for scaling: distributed compute, HPC, cloud-native orchestration. Covers resilience, scheduling, and cost/energy considerations.  

*Suggested Readings:*  
- KubeFlow or Ray documentation.  
- DOE report on AI for Science (2020).  

### Wed Nov 19 — Lecture 16: Automation in Practice [SC week]
Demonstration of automation pipelines with monitoring, logging, and adaptive workflows. Emphasis on debugging and error recovery.  

*Suggested Readings:*  
- MLflow for experiment tracking.  
- Globus Flow for automation.  

---

## No class week of Nov 24 -- Thanksgiving

---

## Week 9 

### Mon Dec 1 — Lecture 17: Frontiers of SDPs
Explores frontiers: multi-agent collaboration, embodied co-scientists, integration with digital twins. Students speculate on SDPs in 2030.  

*Readings:*  
- *[The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search](https://arxiv.org/abs/2504.08066).*  
- Digital twin literature (manufacturing & climate).  

### Wed Dec 3 — Lecture 18: Capstone Prep + Peer Review
Students present draft capstone plans, receive structured peer critique, and refine. Instructor provides guidance on scope, deliverables, and evaluation.  

*Suggested Readings:*  
- Project management frameworks (Agile for research).  
- Sample capstone projects from ML/AI courses.  

---

## Final Week

- **Mon Dec 8 — No Class**  
- **Wed Dec 10 — Final Class Meeting: Capstone Presentations**  

---

