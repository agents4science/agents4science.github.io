from ..base_agent import Tool

@Tool
async def provenance_tool(record: dict):
    rid = record.get("id", "unknown")
    return {"status": "recorded", "uri": f"prov://{rid}"}
