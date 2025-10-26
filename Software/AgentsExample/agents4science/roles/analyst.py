from ..base_agent import Agent
from ..tools.analysis import analysis_tool

class AnalystAgent(Agent):
    def __init__(self, model):
        super().__init__(model=model, tools=[analysis_tool], name="Analyst")

    async def act(self, dataset_ref: str):
        data = await self.tools[0](dataset_ref)
        result = await self.ask(f"Summarize dataset and estimate uncertainty:\n{data}")
        return result
