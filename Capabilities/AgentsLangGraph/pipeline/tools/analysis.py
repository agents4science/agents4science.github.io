from langchain.tools import tool
import numpy as np

@tool("analyze_dataset", return_direct=False)
def analyze_dataset(dataset_ref: str) -> str:
    """Compute mean and std for a mock dataset; returns a short text summary."""
    rng = np.random.default_rng(42)
    data = rng.normal(0.5, 0.05, size=200)
    return f"Dataset {dataset_ref}: mean={data.mean():.4f}, std={data.std():.4f}, n={data.size}"
