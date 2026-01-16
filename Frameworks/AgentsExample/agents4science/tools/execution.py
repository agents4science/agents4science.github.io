
from ..base_agent import Tool

@Tool
async def execution_tool(task):
    return {"status": "submitted", "job_id": "abc123", "bytes": len(task) if isinstance(task, str) else 0}
