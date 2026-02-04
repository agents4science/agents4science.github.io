# API Agent Example

An agent that queries external scientific APIs to retrieve chemical information.

**Code:** [github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsAPI](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsAPI)

## What It Does

1. User asks about a chemical compound
2. Agent calls PubChem API to search for the compound
3. Agent retrieves properties (molecular weight, formula, etc.)
4. Agent synthesizes a response with the retrieved information

## The Code

```python
@tool
def search_compound(name: str) -> str:
    """Search PubChem for a compound by name, returns CID (compound ID)."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
    response = requests.get(url)
    return response.json()

@tool
def get_compound_properties(cid: int) -> str:
    """Get properties for a compound by its PubChem CID."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/..."
    response = requests.get(url)
    return response.json()

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, [search_compound, get_compound_properties])
```

## Running the Example

```bash
cd Capabilities/local-agents/AgentsAPI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Custom query:

```bash
python main.py --query "What is the molecular weight of aspirin?"
```

## LLM Configuration

Supports OpenAI, FIRST (HPC inference), Ollama (local), or mock mode.

See [LLM Configuration](/Capabilities/local-agents/llm-configuration/) for details on configuring LLM backends, including Argonne's FIRST service.

## PubChem API

This example uses the [PubChem PUG REST API](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest), which is:
- Free to use
- No authentication required
- Extensive chemical database (100M+ compounds)

## Mock Data

The `data/` directory contains fallback compound data for mock mode or when the API is unavailable:

```
data/
└── compounds.json    # Aspirin, caffeine, glucose, ethanol, methane, CO2
```

Add more compounds to `compounds.json` to extend the mock data.

## Key Points

- **API chaining**: Agent first searches by name, then fetches properties by ID
- **External data**: Real-time access to PubChem's chemical database
- **Fallback**: Uses local `compounds.json` when API is unavailable

## Requirements

- Python 3.10+
- LangGraph 1.0+
- requests
- OpenAI API key, FIRST token, Ollama, or run in mock mode
