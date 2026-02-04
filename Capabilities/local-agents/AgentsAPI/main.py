"""
API Agent: Query external scientific APIs (PubChem) for chemical information.

Supports four modes:
1. OPENAI_API_KEY set -> uses OpenAI
2. FIRST_API_KEY set -> uses FIRST (HPC inference service)
3. OLLAMA_MODEL set -> uses Ollama (local LLM)
4. None of the above -> uses mock responses to demonstrate the pattern
"""

import argparse
import json
import os
from pathlib import Path

import requests
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# Data directory containing mock compound data
DATA_DIR = Path(__file__).parent / "data"
COMPOUNDS_FILE = DATA_DIR / "compounds.json"

# Load mock data for common compounds (used in mock mode or as fallback)
print(f"Loading compound data from {COMPOUNDS_FILE.name}...")
with open(COMPOUNDS_FILE) as f:
    MOCK_COMPOUNDS = json.load(f)


def get_llm():
    """Get the appropriate LLM based on available credentials."""
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return (
            ChatOpenAI(model="gpt-4o-mini"),
            "OpenAI",
            "OPENAI_API_KEY found in environment",
        )

    if os.environ.get("FIRST_API_KEY"):
        from langchain_openai import ChatOpenAI
        model = os.environ.get("FIRST_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
        base_url = os.environ.get("FIRST_API_BASE", "https://api.first.example.com/v1")
        return (
            ChatOpenAI(model=model, api_key=os.environ["FIRST_API_KEY"], base_url=base_url),
            "FIRST",
            f"FIRST_API_KEY found in environment (model: {model})",
        )

    if os.environ.get("OLLAMA_MODEL"):
        from langchain_openai import ChatOpenAI
        model = os.environ["OLLAMA_MODEL"]
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        base_url = f"{host}/v1"
        return (
            ChatOpenAI(model=model, api_key="ollama", base_url=base_url),
            "Ollama",
            f"OLLAMA_MODEL found in environment (model: {model})",
        )

    return (
        None,
        "Mock",
        "No API key or OLLAMA_MODEL found; using hardcoded responses",
    )


def print_mode_info(mode: str, reason: str):
    """Print information about the selected LLM mode."""
    print("=" * 60)
    print(f"LLM Mode: {mode}")
    print(f"  Reason: {reason}")
    print("=" * 60)


@tool
def search_compound(name: str) -> str:
    """
    Search PubChem for a compound by name.
    Returns the PubChem Compound ID (CID) if found.
    """
    # Check mock data first (for common compounds or when API fails)
    name_lower = name.lower()
    if name_lower in MOCK_COMPOUNDS:
        cid = MOCK_COMPOUNDS[name_lower]["cid"]
        return f"Found compound '{name}' with PubChem CID: {cid}"

    # Try PubChem API
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            if cids:
                return f"Found compound '{name}' with PubChem CID: {cids[0]}"
        return f"Compound '{name}' not found in PubChem"
    except Exception as e:
        # Fall back to mock data
        return f"API error: {e}. Compound '{name}' not found in local cache."


@tool
def get_compound_properties(cid: int) -> str:
    """
    Get properties for a compound by its PubChem CID.
    Returns molecular formula, weight, IUPAC name, and other properties.
    """
    # Check mock data first
    for compound in MOCK_COMPOUNDS.values():
        if compound["cid"] == cid:
            return f"""Properties for CID {cid}:
- Name: {compound['name']}
- Molecular Formula: {compound['formula']}
- Molecular Weight: {compound['molecular_weight']} g/mol
- IUPAC Name: {compound['iupac_name']}
- Description: {compound['description']}"""

    # Try PubChem API
    try:
        properties = "MolecularFormula,MolecularWeight,IUPACName"
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{properties}/JSON"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
            return f"""Properties for CID {cid}:
- Molecular Formula: {props.get('MolecularFormula', 'N/A')}
- Molecular Weight: {props.get('MolecularWeight', 'N/A')} g/mol
- IUPAC Name: {props.get('IUPACName', 'N/A')}"""
        return f"Could not retrieve properties for CID {cid}"
    except Exception as e:
        return f"API error: {e}"


@tool
def compare_compounds(names: list[str]) -> str:
    """
    Compare properties of multiple compounds.
    Provide a list of compound names to compare.
    """
    results = []
    for name in names[:5]:  # Limit to 5 compounds
        name_lower = name.lower()
        if name_lower in MOCK_COMPOUNDS:
            c = MOCK_COMPOUNDS[name_lower]
            results.append(f"{c['name']}: {c['formula']}, MW={c['molecular_weight']}")
        else:
            results.append(f"{name}: Not found in database")
    return "Comparison:\n" + "\n".join(results)


def run_with_llm(llm, query: str):
    """Run the API agent with a real LLM."""
    from langgraph.prebuilt import create_react_agent

    print(f"\nQuery: {query}")
    print("-" * 60)

    agent = create_react_agent(llm, [search_compound, get_compound_properties, compare_compounds])

    for step in agent.stream({"messages": [HumanMessage(content=query)]}):
        if "agent" in step:
            msg = step["agent"]["messages"][0]
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc["name"]
                    args = tc["args"]
                    print(f"Agent calls: {tool_name}({args})")
            if msg.content:
                print(f"\nAnswer: {msg.content}")
        elif "tools" in step:
            content = step["tools"]["messages"][0].content
            print(f"Result: {content}\n")


def run_mock(query: str):
    """Demonstrate the API agent pattern with mock responses."""
    print("\nDemonstrating API query pattern with mock responses.")
    print("Set OPENAI_API_KEY to use a real LLM.\n")

    print(f"Query: {query}")
    print("-" * 60)

    query_lower = query.lower()

    # Determine which compound the user is asking about
    compound_name = None
    for name in MOCK_COMPOUNDS:
        if name in query_lower:
            compound_name = name
            break

    if compound_name is None:
        compound_name = "aspirin"  # Default example

    compound = MOCK_COMPOUNDS[compound_name]

    # Simulate agent flow
    print(f"Agent calls: search_compound('{compound_name}')")
    search_result = search_compound.invoke(compound_name)
    print(f"Result: {search_result}\n")

    print(f"Agent calls: get_compound_properties({compound['cid']})")
    props_result = get_compound_properties.invoke(compound["cid"])
    print(f"Result: {props_result}\n")

    # Generate answer based on query type
    if "molecular weight" in query_lower or "weight" in query_lower:
        answer = f"""The molecular weight of {compound['name']} is **{compound['molecular_weight']} g/mol**.

Its molecular formula is {compound['formula']}, which gives us this weight based on:
- Carbon (C): 12.01 g/mol
- Hydrogen (H): 1.008 g/mol
- Oxygen (O): 16.00 g/mol
- Nitrogen (N): 14.01 g/mol (if present)"""

    elif "formula" in query_lower:
        answer = f"""The molecular formula of {compound['name']} is **{compound['formula']}**.

This means each molecule contains:
{_describe_formula(compound['formula'])}"""

    else:
        answer = f"""{compound['name']} (PubChem CID: {compound['cid']}) is {compound['description']}.

Key properties:
- **Molecular Formula**: {compound['formula']}
- **Molecular Weight**: {compound['molecular_weight']} g/mol
- **IUPAC Name**: {compound['iupac_name']}"""

    print(f"Answer: {answer}")


def _describe_formula(formula: str) -> str:
    """Helper to describe a molecular formula."""
    # Simple parsing for common elements
    descriptions = []
    import re
    elements = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    for elem, count in elements:
        if elem:
            count = int(count) if count else 1
            descriptions.append(f"- {count} {elem} atom(s)")
    return "\n".join(descriptions) if descriptions else "- (complex formula)"


def main():
    parser = argparse.ArgumentParser(description="API Agent for chemical information")
    parser.add_argument(
        "--query",
        default="What is the molecular weight of caffeine?",
        help="Question about a chemical compound",
    )
    args = parser.parse_args()

    llm, mode, reason = get_llm()
    print_mode_info(mode, reason)

    if llm:
        run_with_llm(llm, args.query)
    else:
        run_mock(args.query)


if __name__ == "__main__":
    main()
