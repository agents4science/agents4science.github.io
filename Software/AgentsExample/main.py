# main.py  —  Dashboard v3 (Top: status table, Bottom: last prompt/response)
import os
import asyncio
import logging
import textwrap
import yaml

from rich.console import Console
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live

from agents4science.roles.scout import     ScoutAgent
from agents4science.roles.planner import   PlannerAgent
from agents4science.roles.operator import  OperatorAgent
from agents4science.roles.analyst import   AnalystAgent
from agents4science.roles.archivist import ArchivistAgent

console = new_console = Console()
AGENTS = ["Scout", "Planner", "Operator", "Analyst", "Archivist"]

# --- Config toggles via env ---
SHOW_UI   = os.getenv("A4S_UI", "1") == "1"      # set A4S_UI=0 to disable the live UI
<<<<<<< HEAD
QUIET_LOG = os.getenv("A4S_QUIET", "1") == "1"   # set A4S_QUIET=1 to mute logs during run
=======
QUIET_LOG = os.getenv("A4S_QUIET", "1") == "1"   # set A4S_QUIET=0 to show logs during run
>>>>>>> 4fe2c078cc061f4a60f6e8f251fe2b68180daa2c
GOAL_FILE = os.getenv("AUDIOGOAL_SFILE", "agents4science/workflows/goals.yaml")
MODEL     = os.getenv("A4S_MODEL", "openai/gpt-oss-20b")

def load_goals():
    path = GOAL_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Goals file not found: {path}\n"
            "Create it or set A4S_GOALS_FILE=/path/to/goals.yaml"
        )
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    goals = data.get("goals", [])
    if not isinstance(goals, list) or not goals:
        raise ValueError("No 'goals' list found in goals.yaml")
    return goals


def build_status_table(goals, statuses) -> Table:
    """
    Build a table with one row per goal and columns for each agent role.
    statuses is a list of dicts: [{'Scout': '[..]', 'Planner': '...', ...}, ...]
    """
    table = Table(title="[bold]Agentic Science — Status[/bold]", expand=True, pad_edge=True)
    table.add_column("Goal", no_wrap=False, ratio=4, style="bold")
    for role in AGENTS:
        table.add_column(role, justify="center", no_wrap=True)

    for goal, st in zip(goals, statuses):
        cells = [goal] + [st.get(role, "[dim]—") for role in AGENTS]
        table.add_row(*cells)
    return table


def render_io_panel(agent_name: str, prompt: str | None, response: str | None) -> Panel:
    """
    Renders the bottom panel showing the last prompt/response from the most recent agent call.
    Falls back gracefully if the agent didn't populate last_prompt/last_response.
    """
    p = (prompt or "").strip()
    r = (response or "").strip()

    if not p and not r:
        body = Text("Waiting for first LLM call...", style="dim")
    else:
        # Truncate long blobs to keep UI snappy; adjust as you like
        p_show = (p[:1200] + " …") if len(p) > 1200 else p
        r_show = (r[:2000] + " …") if len(r) > 2000 else r

        md = f"### 🧠 {agent_name} — Prompt to {MODEL}\n```\n{p_show}\n```\n\n**Response**\n```\n{r_show}\n```"
        body = Markdown(md)
        print('DISPLAY', p_show, r_show)

    return Panel(body, title="Last LLM Call", border_style="cyan", padding=(1, 2))


async def run_goal(counter: int, goal: str, status_row: dict, live: Live | None, layout: Layout | None,
                   goals: list[str], statuses: list[dict]):
    """
    Run the five role agents in sequence for a single goal.
    After each agent finishes, update the status table (top) and the last call panel (bottom).
    """
    # Instantiate fresh agents per goal
    agents = [
        ("Scout", ScoutAgent(model=MODEL)),
        ("Planner", PlannerAgent(model=MODEL)),
        ("Operator", OperatorAgent(model=MODEL)),
        ("Analyst", AnalystAgent(model=MODEL)),
        ("Archivist", ArchivistAgent(model=MODEL))
    ]

    def refresh_ui(active_agent_name: str | None = None):
        if not live or not layout:
            return

        # --- Update the status table (top) ---
        layout["status"].update(build_status_table(goals, statuses))

        # --- Only replace the conversation panel if a new agent update arrives ---
        if active_agent_name:
            for name, agent in agents:
                if name == active_agent_name:
                    prompt = getattr(agent, "last_prompt", "")
                    response = getattr(agent, "last_response", "")
                    # only update if there's a real response
                    if response or prompt:
                        layout["conversation"].update(
                            render_io_panel(name, prompt, response)
                        )
                    break

        # otherwise, leave the last panel as-is
        live.refresh()

    def refresh_ui_old(active_agent_name: str | None = None):
        if not live or not layout:
            return
        # Update top status table
        layout["status"].update(build_status_table(goals, statuses))
        # Update bottom panel with the last call from the active agent (if provided)
        if active_agent_name:
            # find the active agent object to grab last_prompt/response
            for name, agent in agents:
                if name == active_agent_name:
                    prompt   = getattr(agent, "last_prompt", "")
                    response = getattr(agent, "last_response", "")
                    layout["conversation"].update( render_io_panel(name, prompt, response) )
                    break
        live.refresh()

    # Mark all cells as pending initially
    for role in agents:
        if role not in status_row:
            status_row[role] = "[dim]Pending"

    # Run each agent in sequence, updating UI between steps
    for name, agent in agents:
        status_row[name] = "[yellow]Running"
        refresh_ui(active_agent_name=name)

        # Dispatch to the role's act() with an appropriate input
        if name in ("Scout", "Planner"):
            result = await agent.act(goal)
            await asyncio.sleep(0.05)  # ensure updates propagated

            prompt = getattr(agent, "last_prompt", "")
            response = getattr(agent, "last_response", "")
            if SHOW_UI: layout["conversation"].update(render_io_panel(agent.name, prompt, response))
            if SHOW_UI: live.refresh()

        elif name == "Operator":
            result = await agent.act(str(status_row))   # simple illustrative payload
        elif name == "Analyst":
            result = await agent.act(f"dataset_{counter:03d}")
        else:  # "Archivist"
            result = await agent.act({"id": f"run-{counter:03d}", "status": "ok"})

        # If Agent.ask() populated last_prompt/last_response, show them; otherwise fallback to result text
        if not getattr(agent, "last_response", None) and isinstance(result, dict) and "response" in result:
            agent.last_response = str(result["response"])
        if not getattr(agent, "last_prompt", None):
            # we don't know the exact prompt string without modifying role code;
            # show a helpful synthesized prompt summary as a fallback
            agent.last_prompt = f"(generated by {name}.act(...))"

        # Mark done and refresh UI
        if SHOW_UI: status_row[name] = "[green]Done"
        if SHOW_UI: refresh_ui(active_agent_name=name)


async def main():
    goals = load_goals()
    statuses = [{role: "[dim]Pending" for role in AGENTS} for _ in goals]

    if not SHOW_UI:
        # Headless mode: no screen drawing, just run the pipeline
        if QUIET_LOG:
            logging.disable(logging.CRITICAL)
        for idx, g in enumerate(goals):
            await run_goal(idx, g, statuses[idx], live=None, layout=None, goals=goals, statuses=statuses)
        return

    # Mute normal logging while the live UI is active to avoid flicker
    if QUIET_LOG:
        logging.disable(logging.INFO)

    # Build two-row layout: top status table, bottom prompt/response panel
    layout = Layout()
    layout.split_column(
        Layout(name="status",       ratio=3),
        Layout(name="conversation", ratio=2),
    )
    layout["status"].update(build_status_table(goals, statuses))
    layout["conversation"].update(Panel(Text("Waiting for first LLM call…", style="dim"),
                                        title="Last LLM Call", border_style="cyan"))

    with Live(layout, refresh_per_second=3, console=console) as live:
        for idx, g in enumerate(goals):
            await run_goal(idx, goals[idx], statuses[idx], live=live, layout=layout,
                           goals=goals, statuses=statuses)

    # Restore logging if it was muted
    if QUIET_LOG: logging.disable(logging.NOTSET)

    # Final static view
    console.clear()
    console.print(build_status_table(goals, statuses))
    console.print("[bold green]All goals complete![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
