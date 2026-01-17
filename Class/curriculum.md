![Header images showing scientists](Assets/newer_bar.jpg?raw=true "Title")

# AI Agents for Science Curriculum

The following **curriculum** outlines topics to be covered and readings, and provides the slides presented in class (minus purely administrative material).

Several guest lectures are not included.

---

### Lecture 1: What is an agent?
Introduces AI agents and and the  sense-plan-act-learn loop. Motivates scientific Discovery Platforms (SDPs): AI-native systems that connect reasoning models with scientific resources. 

*Slides:* [Lecture 1 slides](Assets/Lecture_1_web.pdf).

*Readings:*
- *[Exploring Large Language Model based Intelligent Agents: Definitions, Methods, and Prospects](https://arxiv.org/abs/2401.03428)*, Cheng et al. (Arxiv, 2024).
- *[Artificial intelligence and illusions of understanding in scientific research](https://www.nature.com/articles/s41586-024-07146-0)*, Messeri & Crockett (Nature, 2023).
- *[The Shift from Models to Compound AI Systems – The Berkeley Artificial Intelligence Research Blog](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/).*

### Lecture 2: Frontiers of Language Models
Surveys frontier reasoning models: general-purpose LLMs (GPT, Claude), domain-specific foundation models (materials, bio, weather), and hybrids. Covers techniques for eliciting better reasoning: prompting, chain-of-thought, retrieval-augmented generation (RAG), fine-tuning, and tool-augmented reasoning.

*Slides:* [Lecture 2 slides](Assets/Lecture_2_web.pdf).

*Readings:* 
- *[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)*.
- *[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948).*
- *[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629).*

### Lecture 3: Systems for Agents
Discusses architectures and frameworks for building multi-agent systems, with emphasis on inter-agent communication, orchestration, and lifecycle management. 

*Slides:* [Lecture 3 slides](Assets/Lecture_3_web.pdf).

*Readings:*
- *[AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)*.
- *[LangGraph](https://langchain-ai.github.io/langgraph/?ajs_aid=70d48da3-5563-4143-ba79-957377dfcf92)*.
- *[AIOS: LLM Agent Operating System](https://arxiv.org/pdf/2403.16971)*.
 
### Lecture 4: Retrieval Augmented Generation (RAG) and Vector Databases
Covers how to augment reasoning models with external knowledge bases, vector search, and hybrid retrieval methods. 

*Slides:* [Lecture 4 slides](Assets/Lecture_4_web.pdf).

*Readings:*
- *[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)*.
- *[The FAISS library](https://arxiv.org/abs/2401.08281)*.


### Lecture 5: Tool Calling
Introduces methods for invoking external tools from reasoning models. Focus on model context protocol (MCP), schema design, and execution management.

*Slides:* [Lecture 5 slides](Assets/Lecture_5_web.pdf).

*Readings:*  
- *[Introduction - Model Context Protocol](https://modelcontextprotocol.io/).*

### Lecture 6: HPC Systems and Self Driving Labs
How SDPs connect to HPC workflows and experimental labs. Covers distributed coordination, robotics, and federated agents. 

*Slides:* [Lecture 6 slides](Assets/Lecture_6_web.pdf).

*Readings:*  
- *[Self-Driving Laboratories for Chemistry and Materials Science](https://pubs.acs.org/doi/10.1021/acs.chemrev.4c00055)*, *Chemical Reviews*.
- *[Empowering Scientific Workflows with Federated Agents](https://arxiv.org/abs/2505.05428)*.

### Lecture 7: Human–AI Workflows
Explores how scientists and agents collaborate: trust boundaries, interaction design, and debugging.  

*Slides:* [Lecture 7 slides](Assets/Lecture_7_web.pdf).

*Readings:*  
- *[Guidelines for Human-AI Interaction](https://dl.acm.org/doi/10.1145/3290605.3300233)*, Amershi et al. (*CHI*, 2019).  
- *[Interactive Debugging and Steering of Multi-Agent AI Systems](https://dl.acm.org/doi/10.1145/3706598.3713581)* (*CHI*, 2025).

### Lecture 8: AI co-scientists for accelerating scientific discovery

Guest lecture by Dr. Arvind Ramanathan.

*Slides:* [Lecture 8 slides](Assets/Lecture_8_web.pdf).

*Readings:*  
- *[The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies](https://www.nature.com/articles/s41586-023-06456-y)*, *Nature*.  
- *[The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search](https://arxiv.org/abs/2504.08066).*  


### Lecture 9: Human–AI Workflows, continued
Further discussion of how scientists and agents collaborate

*Slides:* [Lecture 9 slides](Assets/Lecture_9_web.pdf).

### Lecture 10: Benchmarking and Evaluation
Frameworks for assessing agents and SDPs: robustness, validity, and relevance.

*Slides:* [Lecture 10 slides](Assets/Lecture_10_web.pdf).

*Readings:*  
- *[MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://arxiv.org/abs/2410.07095).*  
- *[AI Agents That Matter](https://arxiv.org/abs/2407.01502).*  
- *[EAIRA: Establishing a Methodology for Evaluating AI Models as Scientific Research Assistants](https://arxiv.org/abs/2502.20309).*  


### Lecture 11: Failures and Safety
Examines why multi-agent systems fail and methods for safety and guardrails.  

*Slides:* [Lecture 11 slides](Assets/Lecture_11_web.pdf).

*Readings:*  
- *[Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2408.00154).*  
- *[AGrail: A Lifelong Agent Guardrail with Effective and Adaptive Safety Detection](https://arxiv.org/abs/2502.11448).*  
- *[Improve accuracy by adding Automated Reasoning checks in Amazon Bedrock Guardrails](https://aws.amazon.com/bedrock/).*


### Lecture 12: Novelty and Plagiarism
Explores originality, credit, and the risks of plagiarism in AI-generated science. 

*Slides:* [Lecture 12 slides](Assets/Lecture_12_web.pdf).

*Readings:*  
- *[Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers](https://arxiv.org/abs/2304.12519).*  
- *[All That Glitters is Not Novel: Plagiarism in AI Generated Research](https://arxiv.org/pdf/2502.16487).*  

*Assignment A5:* Capstone project planning (novel contributions).  

### Lecture 13: Building Agents and Workflows
Pipelines, workflow composition, and self-improving systems. 

*Slides:* [Lecture 13 slides](Assets/Lecture_13_web.pdf).

*Readings:*  
- *[AFlow: Automating Agentic Workflow Generation](https://arxiv.org/abs/2410.10762).*  
- *[DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714).*  

### Lecture 14: Finetuning
Covers approaches to adapt agents with reinforcement learning and real-world training.  

*Slides:* [Lecture 14 slides](Assets/Lecture_14_web.pdf).

*Readings:*  
- *[OpenPipe/ART: Agent Reinforcement Trainer](https://github.com/OpenPipe/ART).*  
- *[Agent Lightning: Train ANY AI Agents with Reinforcement Learning](https://arxiv.org/abs/2508.03680).*  

