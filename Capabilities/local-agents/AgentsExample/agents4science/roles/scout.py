from ..base_agent import Agent
from ..tools.streams import stream_tool

class ScoutAgent(Agent):
    def __init__(self, model):
        super().__init__(model=model, tools=[stream_tool], name="Scout")

    async def act(self, query: str):
        data = await self.tools[0]()
        result = await self.ask(f"Analyze for anomalies:\n{data}")
        return result
