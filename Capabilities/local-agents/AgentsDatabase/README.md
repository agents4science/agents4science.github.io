# Database Agent Example

An agent that answers questions about scientific data using pandas queries.

**Code:** [github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsDatabase](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsDatabase)

## What It Does

1. Loads a scientific dataset (catalyst experiment results)
2. User asks a natural language question about the data
3. Agent decides what queries to run
4. Agent interprets results and provides an answer

## The Code

```python
@tool
def query_data(query: str) -> str:
    """Execute a pandas query on the catalyst dataset. Use df.query() syntax."""
    result = df.query(query)
    return result.to_string()

@tool
def describe_columns() -> str:
    """Get information about available columns in the dataset."""
    return df.dtypes.to_string()

@tool
def get_statistics(column: str) -> str:
    """Get summary statistics for a numeric column."""
    return df[column].describe().to_string()

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, [query_data, describe_columns, get_statistics])
```

## Running the Example

```bash
cd Capabilities/local-agents/AgentsDatabase
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Custom question:

```bash
python main.py --question "Which catalyst has the highest efficiency?"
```

## LLM Configuration

Supports OpenAI, FIRST (HPC inference), Ollama (local), or mock mode.

See [LLM Configuration](/Capabilities/local-agents/llm-configuration/) for details on configuring LLM backends, including Argonne's FIRST service.

## Sample Dataset

The `data/` directory contains a sample CSV dataset of catalyst experiments:

```
data/
└── catalyst_experiments.csv
```

Columns:
- `catalyst`: Catalyst material (Cu, Ag, Au, Fe-SAC, etc.)
- `temperature_c`: Reaction temperature in Celsius
- `efficiency_pct`: Conversion efficiency percentage
- `product`: Main product (CO, CH4, C2H4, CH3OH)
- `stability_hrs`: Hours of stable operation

To use your own data, replace `catalyst_experiments.csv` with your CSV file.

## Key Points

- **Schema inspection**: Agent can examine available columns before querying
- **Flexible queries**: Natural language translated to pandas operations
- **Statistics**: Agent can compute summaries for numeric columns

## Requirements

- Python 3.10+
- LangGraph 1.0+
- pandas
- OpenAI API key, FIRST token, Ollama, or run in mock mode
