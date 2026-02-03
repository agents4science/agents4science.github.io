"""
AgentsLangChainDashboard - Rich dashboard for the LangChain multi-agent pipeline.

This wraps the simple AgentsLangChain example with a full-screen Rich dashboard
showing live progress across multiple scientific goals.

Usage:
    python main.py
    python main.py --goals goals.yaml
    A4S_UI=0 python main.py  # headless mode
"""

import os
import sys
import argparse
import yaml

from rich.console import Console
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live

# Add parent directory to path so we can import from AgentsLangChain
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "AgentsLangChain"))

from pipeline.roles import build_roles

console = Console()
AGENTS = ["Scout", "Planner", "Operator", "Analyst", "Archivist"]

# Config via environment
SHOW_UI = os.getenv("A4S_UI", "1") == "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LangChain multi-agent pipeline with Rich dashboard."
    )
    parser.add_argument(
        "--goals",
        "-g",
        type=str,
        default="goals.yaml",
        help="Path to goals YAML file",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    return parser.parse_args()


def load_goals(path: str) -> list[str]:
    """Load goals from YAML file."""
    if not os.path.exists(path):
        # Default goals if file not found
        return [
            "Find catalysts that improve CO2 conversion at room temperature.",
            "Design a polymer that biodegrades in marine environments.",
            "Discover materials with high thermal conductivity for electronics.",
        ]
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    goals = data.get("goals", [])
    if not goals:
        raise ValueError(f"No 'goals' list found in {path}")
    return goals


def build_status_table(goals: list[str], statuses: list[dict]) -> Table:
    """Build a table with one row per goal and columns for each agent role."""
    table = Table(
        title="[bold]LangChain Multi-Agent Pipeline[/bold]",
        expand=True,
        pad_edge=True,
    )
    table.add_column("Goal", no_wrap=False, ratio=4, style="bold")
    for role in AGENTS:
        table.add_column(role, justify="center", no_wrap=True)

    for goal, st in zip(goals, statuses):
        # Truncate long goals for display
        goal_display = goal[:60] + "..." if len(goal) > 60 else goal
        cells = [goal_display] + [st.get(role, "[dim]-") for role in AGENTS]
        table.add_row(*cells)
    return table


def render_output_panel(agent_name: str, output: str) -> Panel:
    """Render panel showing the last agent output."""
    if not output:
        body = Text("Waiting for agent output...", style="dim")
    else:
        # Truncate long outputs
        output_show = output[:2000] + " ..." if len(output) > 2000 else output
        body = Text(output_show)

    return Panel(
        body,
        title=f"[bold cyan]{agent_name}[/bold cyan] Output",
        border_style="cyan",
        padding=(1, 2),
    )


def run_goal(
    goal_idx: int,
    goal: str,
    status_row: dict,
    agents: list,
    live: Live | None,
    layout: Layout | None,
    goals: list[str],
    statuses: list[dict],
) -> None:
    """Run the five agents in sequence for a single goal."""

    def refresh_ui(agent_name: str, output: str = ""):
        if not live or not layout:
            return
        layout["status"].update(build_status_table(goals, statuses))
        if agent_name and output:
            layout["output"].update(render_output_panel(agent_name, output))
        live.refresh()

    state = goal
    for agent in agents:
        name = agent.name
        status_row[name] = "[yellow]Running"
        refresh_ui(name)

        # Run the agent
        output = agent.act(state)

        # Update state for next agent
        state = output

        # Mark done
        status_row[name] = "[green]Done"
        refresh_ui(name, output)


def main():
    args = parse_args()
    goals = load_goals(args.goals)
    statuses = [{role: "[dim]Pending" for role in AGENTS} for _ in goals]

    # Build agents once (they're reusable)
    agents = build_roles(model=args.model)

    if not SHOW_UI:
        # Headless mode
        for idx, goal in enumerate(goals):
            print(f"\n{'='*60}")
            print(f"Goal {idx+1}: {goal}")
            print("=" * 60)
            run_goal(idx, goal, statuses[idx], agents, None, None, goals, statuses)
        return

    # Build layout: top status table, bottom output panel
    layout = Layout()
    layout.split_column(
        Layout(name="status", ratio=3),
        Layout(name="output", ratio=2),
    )
    layout["status"].update(build_status_table(goals, statuses))
    layout["output"].update(
        Panel(
            Text("Waiting for first agent...", style="dim"),
            title="Agent Output",
            border_style="cyan",
        )
    )

    with Live(layout, refresh_per_second=4, console=console) as live:
        for idx, goal in enumerate(goals):
            run_goal(idx, goal, statuses[idx], agents, live, layout, goals, statuses)

    # Final static view
    console.clear()
    console.print(build_status_table(goals, statuses))
    console.print("[bold green]All goals complete![/bold green]")


if __name__ == "__main__":
    main()
