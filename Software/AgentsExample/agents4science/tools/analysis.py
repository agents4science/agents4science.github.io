from ..base_agent import Tool
import numpy as np

@Tool
async def analysis_tool(dataset_ref: str):
    rng = np.random.default_rng(7)
    data = rng.normal(0.5, 0.05, size=100)
    return {"mean": float(data.mean()), "std": float(data.std()), "n": int(data.size), "ref": dataset_ref}
