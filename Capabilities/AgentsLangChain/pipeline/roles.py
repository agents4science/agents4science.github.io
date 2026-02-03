"""Agent role definitions for the scientific discovery pipeline."""
from pipeline.agent import LangAgent
from pipeline.tools.analysis import analyze_dataset


def build_roles(model: str | None = None) -> list[LangAgent]:
    """Return the five LangChain agents used in the workflow."""
    return [
        LangAgent(
            "Scout",
            "Detect anomalies and propose opportunities based on input data.",
            model=model,
        ),
        LangAgent(
            "Planner",
            "Design workflows and allocate resources for scientific goals.",
            model=model,
        ),
        LangAgent(
            "Operator",
            "Execute workflows safely and ensure experiment success.",
            model=model,
        ),
        LangAgent(
            "Analyst",
            "Summarize results and analyze datasets; use tools when appropriate.",
            tools=[analyze_dataset],
            model=model,
        ),
        LangAgent(
            "Archivist",
            "Record provenance and ensure reproducibility of scientific results.",
            model=model,
        ),
    ]
