# Agentic Scientific Discovery Platforms (SDPs)

**Fall 2025 Course Syllabus**

**Schedule:** Mondays & Wednesdays, 3:00pm-4:20pm  
**Start Date:** Monday, Sept 29, 2025  
**End Date:** Wednesday, Dec 10, 2025 (Final Class Meeting: Capstone Presentations)  
**No Class:** Week of Nov 24 (Thanksgiving); Monday, Dec 8 (College Reading Period)

The following **draft curriculum** outlines topics to be covered and potential readings. 

---

## Week 1

### Mon Sept 29 — Lecture 1: What is an SDP?
Introduces the concept of Scientific Discovery Platforms (SDPs): AI-native systems that connect reasoning models with scientific resources. We’ll explore motivating case studies (wildfire hazard, antimicrobials, climate modeling) and outline the challenges of integrating AI into rigorous science.  

*Suggested Readings/Tools:*  
- *[Artificial intelligence and illusions of understanding in scientific research](https://www.nature.com/articles/s41586-024-07146-0)*, Messeri & Crockett (Nature, 2023).  
- Globus Compute docs for distributed science workflows.  <!--Wrong pointer?-->

### Wed Oct 1 — Lecture 2: Landscape of Reasoning AI Models
Surveys frontier reasoning models: general-purpose LLMs (GPT, Claude), domain-specific foundation models (materials, bio, weather), and hybrids. We compare their reasoning abilities, biases, and trade-offs in scientific contexts.  

*Suggested Readings/Tools:*  
- *[Capabilities and risks from frontier AI](https://www.gov.uk/government/publications/frontier-ai-capabilities-and-risks-discussion-paper)* (AI Safety Summit, 2024).  
- HuggingFace model zoo (domain FMs).  

---

## Week 2

### Mon Oct 6 — Lecture 3: Engaging Models
Covers techniques for eliciting better reasoning: prompting, chain-of-thought, retrieval-augmented generation (RAG), fine-tuning, and tool-augmented reasoning. Discusses criteria for model selection.

*Suggested Readings/Tools:*  
- Anthropic *Constitutional AI* whitepaper.  
- LangChain / LlamaIndex tutorials.  

### Wed Oct 8 — Lecture 4: Hands-On: Comparing Models
Lab session applying the same scientific question to different models (general vs domain-specific) and analyzing outputs. Emphasis on differences in accuracy, coverage, and reliability.

*Suggested Readings/Tools:*  
- OpenAI API playground or HuggingFace inference endpoints.  
- Sample dataset: PubChem or Materials Project abstracts.  

---

## Week 3

### Mon Oct 13 — Lecture 5: Knowledge Representation for SDPs
Examines ways to structure knowledge for AI reasoning: embeddings, ontologies, knowledge graphs, curated corpora. Discusses challenges of bias, incompleteness, and negative results.  

*Suggested Readings/Tools:*  
- “Knowledge Graphs and AI” (*IEEE Intelligent Systems*, 2022).  
- Semantic Scholar API.  

### Wed Oct 15 — Lecture 6: Building Knowledge Access Pipelines
Demonstration of building semantic retrieval systems: embedding literature, running vector search, integrating with LLMs. Participants build a mini knowledge base for their domain.  

*Suggested Readings/Tools:*  
- FAISS / Milvus vector databases.  
- Example corpus: ArXiv or PubMed subset.  

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

## Week 9  *(No class week of Nov 24)*

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

