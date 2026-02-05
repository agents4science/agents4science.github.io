"""
RAG Agent: Retrieval-Augmented Generation for scientific documents.

Supports four modes:
1. OPENAI_API_KEY set -> uses OpenAI
2. FIRST_API_KEY set -> uses FIRST (HPC inference service)
3. OLLAMA_MODEL set -> uses Ollama (local LLM)
4. None of the above -> uses mock responses to demonstrate the pattern
"""

import argparse
import os
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# Data directory containing sample documents
DATA_DIR = Path(__file__).parent / "data"


def load_documents() -> list[str]:
    """Load all text documents from the data directory."""
    documents = []
    if DATA_DIR.exists():
        for filepath in sorted(DATA_DIR.glob("*.txt")):
            text = filepath.read_text().strip()
            if text:
                documents.append(text)
                print(f"  Loaded: {filepath.name}")
    return documents


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


# Global vectorstore (initialized lazily)
_vectorstore = None


def get_vectorstore():
    """Get or create the vector store with documents from data/ directory."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        print("Loading documents from data/ directory...")
        documents = load_documents()
        if not documents:
            print("  No documents found in data/ directory!")
            return None

        embeddings = OpenAIEmbeddings()
        docs = [Document(page_content=text) for text in documents]
        print("Creating embeddings...")
        _vectorstore = FAISS.from_documents(docs, embeddings)
        return _vectorstore

    # For non-OpenAI modes, return None (will use mock search)
    return None


@tool
def search_documents(query: str) -> str:
    """Search the scientific document collection for relevant passages about CO2 catalysis."""
    vectorstore = get_vectorstore()

    if vectorstore is not None:
        docs = vectorstore.similarity_search(query, k=3)
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(f"[{i}] {doc.page_content}")
        return "\n\n".join(results)

    # Mock search results based on query keywords
    query_lower = query.lower()
    if "room temperature" in query_lower or "ambient" in query_lower:
        return """[1] Room temperature CO2 conversion remains challenging due to kinetic barriers.
Photocatalytic approaches using TiO2 and related semiconductors can drive the
reaction with solar energy, but efficiencies are typically below 1%. Plasma-
assisted catalysis is an emerging approach that can activate CO2 at ambient conditions.

[2] Gold and silver catalysts are highly selective for CO2 reduction to CO,
achieving nearly 100% Faradaic efficiency. These catalysts operate well at room
temperature in aqueous electrolytes."""

    elif "copper" in query_lower or "cu" in query_lower:
        return """[1] Copper-based catalysts have shown promising results for electrochemical CO2
reduction. Cu nanoparticles can achieve Faradaic efficiencies above 60% for
producing multi-carbon products like ethylene and ethanol.

[2] Cu-MOFs have demonstrated activity for CO2 to methanol conversion at
moderate temperatures (150-250C)."""

    elif "challenge" in query_lower or "difficult" in query_lower:
        return """[1] Challenges in CO2 conversion include: (1) the high stability of CO2 molecule
requiring significant energy input, (2) competing hydrogen evolution reaction
in aqueous systems, (3) catalyst deactivation over time, and (4) scaling up
from laboratory to industrial scale.

[2] Room temperature CO2 conversion remains challenging due to kinetic barriers."""

    else:
        return """[1] Copper-based catalysts have shown promising results for electrochemical CO2
reduction. Cu nanoparticles can achieve Faradaic efficiencies above 60%.

[2] Single-atom catalysts (SACs) show improved activity for CO2 reduction.
Iron and nickel SACs on nitrogen-doped carbon supports can selectively produce CO.

[3] Metal-organic frameworks (MOFs) offer tunable pore structures for CO2 capture
and conversion."""


def run_with_llm(llm, question: str):
    """Run the RAG agent with a real LLM."""
    from langchain.agents import create_agent

    print(f"\nQuestion: {question}")
    print("-" * 60)

    # Initialize vectorstore
    get_vectorstore()
    print("Ready.\n")

    agent = create_agent(llm, [search_documents])

    for step in agent.stream({"messages": [HumanMessage(content=question)]}):
        if "agent" in step:
            msg = step["agent"]["messages"][0]
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"Agent searches: {tc['args'].get('query', '')[:60]}...")
            if msg.content:
                print(f"\nAnswer: {msg.content}")
        elif "tools" in step:
            content = step["tools"]["messages"][0].content
            # Show truncated results
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"Retrieved:\n{preview}\n")


def run_mock(question: str):
    """Demonstrate the RAG pattern with mock responses."""
    print("\nDemonstrating RAG pattern with mock responses.")
    print("Set OPENAI_API_KEY to use real embeddings and LLM.\n")

    # Show available documents
    print("Documents in data/ directory:")
    if DATA_DIR.exists():
        for filepath in sorted(DATA_DIR.glob("*.txt")):
            print(f"  - {filepath.name}")
    print()

    print(f"Question: {question}")
    print("-" * 60)

    # Simulate the agent flow
    print("Agent searches: CO2 conversion catalysts...")
    results = search_documents.invoke(question)
    preview = results[:300] + "..." if len(results) > 300 else results
    print(f"Retrieved:\n{preview}\n")

    # Mock answer based on retrieved content
    if "room temperature" in question.lower():
        answer = """Based on the retrieved documents, room temperature CO2 conversion faces
significant kinetic barriers. Current approaches include:

1. **Photocatalysis**: TiO2-based systems can work at ambient conditions but
   have low efficiencies (<1%)
2. **Plasma-assisted catalysis**: An emerging approach that activates CO2
   at ambient conditions
3. **Electrochemical reduction**: Gold and silver catalysts work at room
   temperature but primarily produce CO rather than higher-value products

The main challenge is overcoming the high stability of the CO2 molecule without
high temperatures or pressures."""
    else:
        answer = """Based on the retrieved documents, several catalyst types show promise
for CO2 conversion:

1. **Copper-based catalysts**: Achieve >60% Faradaic efficiency for multi-carbon
   products like ethylene and ethanol
2. **Single-atom catalysts (SACs)**: Iron and nickel SACs on N-doped carbon
   selectively produce CO at low overpotentials
3. **Metal-organic frameworks (MOFs)**: Offer tunable structures; Cu-MOFs can
   convert CO2 to methanol at moderate temperatures

Key challenges include catalyst deactivation and scaling to industrial processes."""

    print(f"Answer: {answer}")


def main():
    parser = argparse.ArgumentParser(description="RAG Agent for scientific documents")
    parser.add_argument(
        "--question",
        default="What catalysts are effective for CO2 conversion at room temperature?",
        help="Question to ask about the documents",
    )
    args = parser.parse_args()

    llm, mode, reason = get_llm()
    print_mode_info(mode, reason)

    if llm:
        run_with_llm(llm, args.question)
    else:
        run_mock(args.question)


if __name__ == "__main__":
    main()
