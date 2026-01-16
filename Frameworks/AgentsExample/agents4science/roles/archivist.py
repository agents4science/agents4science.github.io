from ..base_agent import Agent
from ..tools.provenance import provenance_tool
import json

class ArchivistAgent(Agent):
    def __init__(self, model):
        super().__init__(model=model, tools=[provenance_tool], name="Archivist")

    async def act(self, record: dict):
        result = await self.tools[0](record)
        payload = json.dumps(record, indent=2)
        reply = await self.ask(f"Record result: {result}\nPayload:\n{payload}")
        return reply
