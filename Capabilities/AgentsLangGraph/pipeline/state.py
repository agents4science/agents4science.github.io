"""State definition for the LangGraph scientific pipeline."""
from typing import TypedDict, Annotated
from operator import add


class PipelineState(TypedDict):
    """State passed between agents in the pipeline.

    Attributes:
        goal: The scientific goal being addressed
        scout_output: Research opportunities identified by Scout
        planner_output: Workflow plan designed by Planner
        operator_output: Execution results from Operator
        analyst_output: Analysis summary from Analyst
        archivist_output: Documented provenance from Archivist
        messages: Accumulated messages/logs from the pipeline
    """
    goal: str
    scout_output: str
    planner_output: str
    operator_output: str
    analyst_output: str
    archivist_output: str
    messages: Annotated[list[str], add]
