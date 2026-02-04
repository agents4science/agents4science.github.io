# RAG Agent Example

An agent that retrieves and answers questions from scientific documents using Retrieval-Augmented Generation (RAG).

**Code:** [github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsRAG](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsRAG)

## What It Does

1. Loads scientific documents and creates embeddings
2. User asks a question about the documents
3. Agent retrieves relevant passages
4. Agent synthesizes an answer from the retrieved context

## The Code

```python
@tool
def search_documents(query: str) -> str:
    """Search the document collection for relevant passages."""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in docs])

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, [search_documents])
agent.invoke({"messages": [HumanMessage(content="What catalysts work for CO2 conversion?")]})
```

## Running the Example

```bash
cd Capabilities/local-agents/AgentsRAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Custom question:

```bash
python main.py --question "What are the challenges with room temperature catalysis?"
```

## LLM Configuration

Supports OpenAI, FIRST (HPC inference), Ollama (local), or mock mode.

See [LLM Configuration](/Capabilities/local-agents/llm-configuration/) for details on configuring LLM backends, including Argonne's FIRST service.

## Sample Data

The example includes sample text about CO2 conversion catalysts. In a real application, you would load your own documents (PDFs, papers, etc.).

## Key Points

- **Vector store**: FAISS stores document embeddings for similarity search
- **Embeddings**: OpenAI embeddings (or mock embeddings in mock mode)
- **Retrieval**: `search_documents` tool finds relevant passages
- **Generation**: LLM synthesizes answer from retrieved context

## Requirements

- Python 3.10+
- LangGraph 1.0+
- FAISS (faiss-cpu)
- OpenAI API key, FIRST token, Ollama, or run in mock mode
