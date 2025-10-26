from ..base_agent import Tool
import random

@Tool
async def stream_tool():
    spectra = [round(1.0 + random.uniform(-0.1, 0.1), 3) for _ in range(5)]
    timestamps = [f"t{i}" for i in range(5)]
    return {"spectra": spectra, "timestamps": timestamps}
