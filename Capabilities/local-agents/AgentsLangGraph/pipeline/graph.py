"""LangGraph workflow for the scientific discovery pipeline."""
from langgraph.graph import StateGraph, START, END

from .state import PipelineState
from .nodes import (
    scout_node,
    planner_node,
    operator_node,
    analyst_node,
    archivist_node,
)


def build_graph() -> StateGraph:
    """Build and compile the scientific discovery pipeline graph.

    The graph implements a linear flow through five specialized agents:
    Scout -> Planner -> Operator -> Analyst -> Archivist

    Returns:
        A compiled StateGraph ready for execution.
    """
    # Create the graph with our state schema
    graph = StateGraph(PipelineState)

    # Add nodes for each agent
    graph.add_node("scout", scout_node)
    graph.add_node("planner", planner_node)
    graph.add_node("operator", operator_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("archivist", archivist_node)

    # Define the linear flow: START -> Scout -> Planner -> Operator -> Analyst -> Archivist -> END
    graph.add_edge(START, "scout")
    graph.add_edge("scout", "planner")
    graph.add_edge("planner", "operator")
    graph.add_edge("operator", "analyst")
    graph.add_edge("analyst", "archivist")
    graph.add_edge("archivist", END)

    # Compile the graph
    return graph.compile()


def run_pipeline(goal: str, verbose: bool = True) -> PipelineState:
    """Run the scientific discovery pipeline for a given goal.

    Args:
        goal: The scientific goal to address
        verbose: Whether to print progress messages

    Returns:
        The final pipeline state with all agent outputs
    """
    app = build_graph()

    # Initialize state with the goal
    initial_state: PipelineState = {
        "goal": goal,
        "scout_output": "",
        "planner_output": "",
        "operator_output": "",
        "analyst_output": "",
        "archivist_output": "",
        "messages": [],
    }

    # Run the graph
    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting pipeline for goal:")
        print(f"  {goal}")
        print(f"{'='*60}\n")

    # Stream execution to see progress
    for event in app.stream(initial_state):
        if verbose:
            # Print which node just completed
            for node_name in event:
                if node_name != "__end__":
                    print(f"=== {node_name.title()} completed ===")
                    output_key = f"{node_name}_output"
                    if output_key in event[node_name]:
                        output = event[node_name][output_key]
                        # Print truncated output
                        print(f"{output[:400]}...\n" if len(output) > 400 else f"{output}\n")

    # Get final state
    final_state = app.invoke(initial_state)
    return final_state
