"""Agent nodes for the LangGraph scientific pipeline."""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .state import PipelineState
from .tools.analysis import analyze_dataset


def _create_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    """Create a ChatOpenAI instance."""
    return ChatOpenAI(model=model, temperature=temperature)


def _create_chain(system_prompt: str, model: str = "gpt-4o-mini"):
    """Create a simple LLM chain with a system prompt."""
    llm = _create_llm(model)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return prompt | llm | StrOutputParser()


def scout_node(state: PipelineState) -> dict:
    """Scout agent: Surveys problem space, detects anomalies, identifies opportunities."""
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
    chain = _create_chain(
        "You are the Analyst agent. Summarize the execution results, "
        "quantify uncertainty, and provide key insights. Include statistical "
        "analysis where appropriate. Reference specific metrics and findings."
    )

    # Optionally call the analysis tool
    tool_result = analyze_dataset.invoke("execution_results")

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
