"""
Agent nodes for the LangGraph scientific pipeline.

Supports three modes:
1. OPENAI_API_KEY set → uses OpenAI
2. FIRST_API_KEY set → uses FIRST (HPC inference service)
3. Neither set → uses mock responses to demonstrate the pattern
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .state import PipelineState
from .tools.analysis import analyze_dataset
from .llm import get_llm, get_llm_mode


def _create_chain(system_prompt: str, model: str = "gpt-4o-mini"):
    """Create a simple LLM chain with a system prompt."""
    llm = get_llm(model)
    if llm is None:
        return None

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return prompt | llm | StrOutputParser()


# Mock responses for demonstration when no API key is available
MOCK_RESPONSES = {
    "scout": """Research Opportunities Identified:

1. **Metal-organic frameworks (MOFs)** - Promising class of materials with tunable pore structures
   that could enable room-temperature CO2 conversion through confinement effects.

2. **Plasma-assisted catalysis** - Non-thermal plasma can activate CO2 at ambient temperatures,
   potentially enabling conversion without traditional thermal requirements.

3. **Electrochemical reduction** - Recent advances in copper-based electrocatalysts show
   improved selectivity for valuable products (ethylene, ethanol) at low overpotentials.

Key anomaly detected: Several recent papers report unexpectedly high activity for
nitrogen-doped carbon catalysts, warranting further investigation.""",

    "planner": """Workflow Plan:

**Phase 1: Literature Review** (Priority: High)
- Systematic search of recent publications on room-temperature CO2 catalysis
- Focus on MOFs, plasma catalysis, and electrochemical approaches
- Resources: Access to scientific databases, text mining tools

**Phase 2: Computational Screening** (Priority: High)
- DFT calculations to identify promising catalyst candidates
- Screen binding energies and activation barriers
- Resources: HPC allocation (estimated 10,000 core-hours)

**Phase 3: Experimental Validation** (Priority: Medium)
- Synthesize top 5 candidates from computational screening
- Characterize using XRD, BET, XPS
- Test catalytic activity under standard conditions

**Phase 4: Optimization** (Priority: Low, contingent on Phase 3 success)
- Optimize synthesis conditions for best-performing catalyst
- Scale-up feasibility assessment""",

    "operator": """Execution Results:

**Phase 1 Complete:**
- Reviewed 247 papers from 2020-2024
- Identified 12 high-potential catalyst systems
- Created annotated bibliography

**Phase 2 In Progress:**
- Submitted 45 DFT calculations to HPC queue
- 32/45 completed successfully
- Preliminary results show Cu-N-C materials most promising

**Issues Encountered:**
- 3 calculations failed due to convergence issues (resubmitted with adjusted parameters)
- One promising MOF structure lacks complete crystallographic data

**Safety Notes:**
- All computational jobs running within allocation limits
- No hazardous materials involved at this stage""",

    "analyst": """Analysis Summary:

**Key Findings:**
1. Cu-N-C catalysts show 3.2x higher predicted activity than baseline Cu catalysts
2. Optimal N-doping level: 4-6 atomic percent
3. Binding energy correlation (R² = 0.87) suggests descriptor-based screening viable

**Statistical Analysis:**
- Mean CO2 conversion efficiency (predicted): 34.2% ± 5.7%
- Confidence interval for top candidate: [31.5%, 42.1%]
- p-value for improvement over baseline: 0.003

**Uncertainty Quantification:**
- DFT calculation uncertainty: ±0.15 eV (typical for binding energies)
- Model extrapolation to experimental conditions adds ~20% uncertainty
- Recommended experimental validation for top 3 candidates

**Recommendations:**
Proceed with synthesis of Cu-N-C materials with 5% N-doping as primary target.""",

    "archivist": """Research Documentation for Reproducibility

**Project:** Room-Temperature CO2 Conversion Catalyst Discovery
**Date:** 2024-01-15
**Status:** Phase 2 Complete

## Methodology
1. Literature mining using Semantic Scholar API (247 papers analyzed)
2. DFT calculations using VASP 6.3.0 with PBE functional
3. Statistical analysis using Python 3.11 + scipy 1.11

## Data Sources
- Cambridge Structural Database for MOF structures
- Materials Project for baseline catalyst properties
- In-house calculations stored at /data/co2_catalyst_screening/

## Key Parameters
- DFT settings: ENCUT=520, KPOINTS=4x4x1, EDIFF=1E-6
- Binding energy reference: gas-phase CO2 at 298K

## Provenance
- All scripts version-controlled in git repository
- Random seeds documented for reproducibility
- HPC job IDs logged for audit trail

## Findings Summary
Top candidate: Cu-N-C with 5% N-doping
Predicted improvement: 3.2x over baseline
Confidence: High (p < 0.01)""",
}


def scout_node(state: PipelineState) -> dict:
    """Scout agent: Surveys problem space, detects anomalies, identifies opportunities."""
    if get_llm_mode() == "mock":
        output = MOCK_RESPONSES["scout"]
    else:
        chain = _create_chain(
            "You are the Scout agent. Survey the problem space, detect anomalies, "
            "and identify research opportunities based on the given scientific goal. "
            "Be concise but thorough in identifying key research directions."
        )
        output = chain.invoke({"input": state["goal"]})

    return {
        "scout_output": output,
        "messages": [f"Scout: Identified opportunities for '{state['goal'][:50]}...'"]
    }


def planner_node(state: PipelineState) -> dict:
    """Planner agent: Designs workflows, allocates resources."""
    if get_llm_mode() == "mock":
        output = MOCK_RESPONSES["planner"]
    else:
        chain = _create_chain(
            "You are the Planner agent. Based on the research opportunities identified, "
            "design a detailed workflow and allocate resources. Include specific steps, "
            "required resources, and timeline considerations."
        )
        input_text = f"Goal: {state['goal']}\n\nOpportunities: {state['scout_output']}"
        output = chain.invoke({"input": input_text})

    return {
        "planner_output": output,
        "messages": ["Planner: Workflow designed"]
    }


def operator_node(state: PipelineState) -> dict:
    """Operator agent: Executes the planned workflow safely."""
    if get_llm_mode() == "mock":
        output = MOCK_RESPONSES["operator"]
    else:
        chain = _create_chain(
            "You are the Operator agent. Execute the planned workflow safely. "
            "Report on execution status, any issues encountered, and results obtained. "
            "Ensure all safety protocols are followed."
        )
        input_text = f"Plan: {state['planner_output']}"
        output = chain.invoke({"input": input_text})

    return {
        "operator_output": output,
        "messages": ["Operator: Workflow executed"]
    }


def analyst_node(state: PipelineState) -> dict:
    """Analyst agent: Summarizes findings, quantifies uncertainty."""
    # The Analyst can use tools - here we demonstrate with analyze_dataset
    tool_result = analyze_dataset.invoke("execution_results")

    if get_llm_mode() == "mock":
        output = MOCK_RESPONSES["analyst"]
    else:
        chain = _create_chain(
            "You are the Analyst agent. Summarize the execution results, "
            "quantify uncertainty, and provide key insights. Include statistical "
            "analysis where appropriate. Reference specific metrics and findings."
        )
        input_text = (
            f"Execution Results: {state['operator_output']}\n\n"
            f"Statistical Analysis: {tool_result}"
        )
        output = chain.invoke({"input": input_text})

    return {
        "analyst_output": output,
        "messages": ["Analyst: Analysis completed"]
    }


def archivist_node(state: PipelineState) -> dict:
    """Archivist agent: Documents everything for reproducibility."""
    if get_llm_mode() == "mock":
        output = MOCK_RESPONSES["archivist"]
    else:
        chain = _create_chain(
            "You are the Archivist agent. Document the entire research process "
            "for reproducibility. Create a structured record including methodology, "
            "data sources, analysis steps, and key findings. Ensure provenance is clear."
        )
        input_text = (
            f"Goal: {state['goal']}\n\n"
            f"Opportunities: {state['scout_output']}\n\n"
            f"Plan: {state['planner_output']}\n\n"
            f"Results: {state['operator_output']}\n\n"
            f"Analysis: {state['analyst_output']}"
        )
        output = chain.invoke({"input": input_text})

    return {
        "archivist_output": output,
        "messages": ["Archivist: Documentation complete"]
    }
