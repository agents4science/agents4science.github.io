# Agentic Scientific Discovery Platforms

The following **draft curriculum** outlines topics to be covered and potential readings. 

---

## Week 1

### Mon Sept 29 — Lecture 1: What is an SDP?
Introduces the concept of Scientific Discovery Platforms (SDPs): AI-native systems that connect reasoning models with scientific resources. We’ll explore motivating case studies (wildfire hazard, antimicrobials, climate modeling) and outline the challenges of integrating AI into rigorous science.  

*Readings:*
- *[Exploring Large Language Model based Intelligent Agents: Definitions, Methods, and Prospects](https://arxiv.org/abs/2401.03428)*, Cheng et al. (Arxiv, 2024).
- *[Artificial intelligence and illusions of understanding in scientific research](https://www.nature.com/articles/s41586-024-07146-0)*, Messeri & Crockett (Nature, 2023).
- *[The Shift from Models to Compound AI Systems – The Berkeley Artificial Intelligence Research Blog](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/).*

### Wed Oct 1 — Lecture 2: Frontiers of Language Models
Surveys frontier reasoning models: general-purpose LLMs (GPT, Claude), domain-specific foundation models (materials, bio, weather), and hybrids. Covers techniques for eliciting better reasoning: prompting, chain-of-thought, retrieval-augmented generation (RAG), fine-tuning, and tool-augmented reasoning.

*Readings:* 
- *[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)*.
- *[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948).*
- *[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629).*

*Assignment A1:* Implement a ReACT style agent.  

---

## Week 2

### Mon Oct 6 — Lecture 3: Systems for Agents
Overview TBD.

*Readings:*
- *[AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)*.
- *[LangGraph](https://langchain-ai.github.io/langgraph/?ajs_aid=70d48da3-5563-4143-ba79-957377dfcf92)*.
- *[AIOS: LLM Agent Operating System](https://arxiv.org/pdf/2403.16971)*.
 
### Wed Oct 8 — Lecture 4: Retrieval Augmented Generation (RAG) and Vector Databases
Overview TBD.

*Readings:*
- *[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)*.
- *[The FAISS library](https://arxiv.org/abs/2401.08281)*.

*Assignment A2:* Hybrid retrieval.  

---

## Week 3

### Mon Oct 13 — Lecture 5: Tool Calling
Overview TBD.

*Readings:*  
- *[Introduction - Model Context Protocol]()*.

### Wed Oct 15 — Lecture 6: HPC Systems and Self Driving Labs
Overview TBD.

*Readings:*  
- *[Self-Driving Laboratories for Chemistry and Materials Science]()*, Chemical Reviews.
- *[Empowering Scientific Workflows with Federated Agents]()*.

*Assignment A3:* Implement Distributed Battleship (and/or Implement MCP toolbox),

---

## Week 4

### Mon Oct 20 — Lecture 7: Connecting to Codes and Labs
Introduces methods to link reasoning models to simulations, databases, and instruments. Discusses APIs, middleware (MCP, Globus Compute), and orchestration engines.  

*Suggested Readings/Tools:*  
- MCP protocol specification.  
- Snakemake or Airflow tutorials.  

### Wed Oct 22 — Lecture 8: Case Studies in Resource Integration
Examines real-world SDP integrations: wildfire simulations, catalyst discovery, automated labs. Participants dissect one workflow end-to-end.  

*Suggested Readings/Tools:*  
- Scientific workflows in PNAS (*Self-driving labs for materials*, 2021).  
- Materials Acceleration Platforms case studies.  

---

## Week 5

### Mon Oct 27 — Lecture 9: The Scientist in the Loop
Explores the evolving role of scientists in SDPs: what to delegate, what to oversee, and how to structure collaboration. Introduces interpretability, failure modes, and cognitive biases.  

*Suggested Readings/Tools:*  
- Amershi et al., “Guidelines for Human-AI Interaction” (*CHI*, 2019).  
- Case study: AlphaFold’s integration into biology.  

### Wed Oct 29 — Lecture 10: Designing Human–AI Workflows
Participants map their domain problems into human vs SDP responsibilities. Exercises focus on trust boundaries, verification strategies, and failure recovery.  

*Suggested Readings/Tools:*  
- STS literature on trust in automation.  
- Example: Radiology AI workflow studies.  

---

## Week 6

### Mon Nov 3 — Lecture 11: Scientific Evaluation
Discusses scientific rigor in SDPs: falsifiability, reproducibility, uncertainty quantification. Identifies pitfalls like “hallucinations” and overfitting to conventions. 

*Suggested Readings/Tools:*  
- Messeri & Crockett, *Nature* (2023).  
- Tutorials on uncertainty estimation in ML.  

### Wed Nov 5 — Lecture 12: Designing Benchmarks for SDPs
Participants design evaluation protocols for their prototypes: from unit-level tests to end-to-end validation. Discusses benchmark suites and shared tasks.  

*Suggested Readings/Tools:*  
- [*EAIRA: Establishing a Methodology for Evaluating AI Models as Scientific Research Assistants*](https://arxiv.org/abs/2502.20309), Cappello et al., 2025.
- MLCommons Scientific Benchmarks.  
- OpenCatalyst Project evaluation suite.  

---

## Week 7

### Mon Nov 10 — Lecture 13: Sensitive & Proprietary Data
Covers privacy-preserving ML (federated learning, differential privacy), governance, and secure computation. Focus on biomedical and environmental domains.  

*Suggested Readings/Tools:*  
- Google AI: *Federated Learning at Scale* (2021).  
- PySyft for privacy-preserving ML.  

### Wed Nov 12 — Lecture 14: Responsible SDPs
Discusses ethical and policy dimensions: dual-use concerns, bias, carbon footprint, open science vs IP. Students draft a responsible-use statement.  

*Suggested Readings/Tools:*  
- UNESCO AI Ethics framework.  
- Carbon-aware computing literature.  

---

## Week 8

### Mon Nov 17 — Lecture 15: Scaling SDPs
Strategies for scaling: distributed compute, HPC, cloud-native orchestration. Covers resilience, scheduling, and cost/energy considerations.  

*Suggested Readings/Tools:*  
- KubeFlow or Ray documentation.  
- DOE report on AI for Science (2020).  

### Wed Nov 19 — Lecture 16: Automation in Practice
Demonstration of automation pipelines with monitoring, logging, and adaptive workflows. Emphasis on debugging and error recovery.  

*Suggested Readings/Tools:*  
- MLflow for experiment tracking.  
- Globus Flow for automation.  

---

## No class week of Nov 24 -- Thanksgiving

---

## Week 9 

### Mon Dec 1 — Lecture 17: Frontiers of SDPs
Explores frontiers: multi-agent collaboration, embodied co-scientists, integration with digital twins. Students speculate on SDPs in 2030.  

*Suggested Readings/Tools:*  
- [The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search](https://github.com/SakanaAI/AI-Scientist-v2) (arXiv, 2024).  
- Digital twin literature (e.g., manufacturing & climate).  

### Wed Dec 3 — Lecture 18: Capstone Prep + Peer Review
Students present draft capstone plans, receive structured peer critique, and refine. Instructor provides guidance on scope, deliverables, and evaluation.  
<!--Too late in the quarter-->

*Suggested Readings/Tools:*  
- Project management frameworks (Agile for research).  
- Sample capstone projects from ML/AI courses.  

---

## Final Week

- **Mon Dec 8 — No Class**  
- **Wed Dec 10 — Final Class Meeting: Capstone Presentations**  

---

