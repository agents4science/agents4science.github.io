from ..base_agent import Agent
from ..tools.execution import execution_tool

class OperatorAgent(Agent):
    def __init__(self, model):
        super().__init__(model=model, tools=[execution_tool], name="Operator")

    async def act(self, workflow_json: str):
        result = await self.tools[0](workflow_json)
        reply = await self.ask(f"Executed workflow result:\n{result}")
        return reply
