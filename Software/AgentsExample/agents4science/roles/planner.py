from ..base_agent import Agent
from ..tools.registry import registry_tool

class PlannerAgent(Agent):
    def __init__(self, model):
        super().__init__(model=model, tools=[registry_tool], name="Planner")

    async def act(self, goal: str):
        available = await self.tools[0]()
        result = await self.ask(f"Available tools: {available}\nDesign a plan for goal:\n{goal}")
        return result
