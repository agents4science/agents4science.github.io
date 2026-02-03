from ..base_agent import Tool

@Tool
async def registry_tool():
    return ["DFT_simulator", "XANES_lab", "HPC_Queue"]
