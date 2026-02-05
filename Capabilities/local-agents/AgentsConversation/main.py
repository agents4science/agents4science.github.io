#!/usr/bin/env python3
"""
Conversational Agent with Memory

Demonstrates memory patterns for stateful conversations:
- Short-term memory: Recent conversation history
- Long-term memory: Persistent facts extracted from conversation
- Memory window: Sliding window to manage context length
- Conversation summary: Compress old messages to stay within limits

Run interactively:
    python main.py

Or with initial context:
    python main.py --context "I'm researching copper catalysts for CO2 reduction"
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

LLM_MODE = None

if os.environ.get("OPENAI_API_KEY"):
    LLM_MODE = "openai"
elif os.environ.get("FIRST_API_KEY"):
    LLM_MODE = "first"
elif os.environ.get("OLLAMA_MODEL"):
    LLM_MODE = "ollama"
else:
    LLM_MODE = "mock"


def get_llm():
    """Get configured LLM based on available credentials."""
    if LLM_MODE == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini")
    elif LLM_MODE == "first":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            base_url="https://api.first.argonne.gov/v1",
            api_key=os.environ["FIRST_API_KEY"]
        )
    elif LLM_MODE == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.environ.get("OLLAMA_MODEL", "llama3.2"))
    else:
        return None


# ============================================================================
# MEMORY SYSTEM
# ============================================================================

@dataclass
class ConversationMemory:
    """Manages both short-term and long-term memory for conversations."""

    # Short-term: recent messages
    messages: list = field(default_factory=list)
    max_messages: int = 20  # Keep last N messages

    # Long-term: extracted facts
    facts: dict = field(default_factory=dict)

    # Conversation summary (for compressing old messages)
    summary: str = ""

    # Session metadata
    session_id: str = ""
    started_at: str = ""

    def __post_init__(self):
        if not self.session_id:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.started_at:
            self.started_at = datetime.now().isoformat()

    def add_message(self, role: str, content: str):
        """Add a message to short-term memory."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Trim if exceeding max
        if len(self.messages) > self.max_messages:
            # Keep system messages and recent messages
            self._compress_old_messages()

    def _compress_old_messages(self):
        """Compress old messages into summary to manage context length."""
        if len(self.messages) <= self.max_messages // 2:
            return

        # Keep most recent half
        keep_count = self.max_messages // 2
        old_messages = self.messages[:-keep_count]
        self.messages = self.messages[-keep_count:]

        # Add old messages to summary
        old_text = "\n".join([f"{m['role']}: {m['content']}" for m in old_messages])
        if self.summary:
            self.summary = f"{self.summary}\n\nEarlier in conversation:\n{old_text}"
        else:
            self.summary = f"Earlier in conversation:\n{old_text}"

    def add_fact(self, key: str, value: str):
        """Store a fact in long-term memory."""
        self.facts[key] = {
            "value": value,
            "added_at": datetime.now().isoformat()
        }

    def get_fact(self, key: str) -> Optional[str]:
        """Retrieve a fact from long-term memory."""
        if key in self.facts:
            return self.facts[key]["value"]
        return None

    def get_all_facts(self) -> dict:
        """Get all stored facts."""
        return {k: v["value"] for k, v in self.facts.items()}

    def get_context_for_llm(self) -> list:
        """Get messages formatted for LLM context."""
        context = []

        # Add summary if exists
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Conversation summary:\n{self.summary}"
            })

        # Add facts if any
        if self.facts:
            facts_text = "\n".join([f"- {k}: {v['value']}" for k, v in self.facts.items()])
            context.append({
                "role": "system",
                "content": f"Known facts about this conversation:\n{facts_text}"
            })

        # Add recent messages
        context.extend(self.messages)

        return context

    def save(self, filepath: Path):
        """Save memory to disk for persistence."""
        data = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "messages": self.messages,
            "facts": self.facts,
            "summary": self.summary
        }
        filepath.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, filepath: Path) -> "ConversationMemory":
        """Load memory from disk."""
        if not filepath.exists():
            return cls()
        data = json.loads(filepath.read_text())
        memory = cls(
            session_id=data.get("session_id", ""),
            started_at=data.get("started_at", ""),
            messages=data.get("messages", []),
            facts=data.get("facts", {}),
            summary=data.get("summary", "")
        )
        return memory


# Global memory instance
MEMORY = ConversationMemory()


# ============================================================================
# MEMORY TOOLS
# ============================================================================

@tool
def remember_fact(key: str, value: str) -> str:
    """Store an important fact for future reference.

    Use this to remember things the user tells you that might be
    relevant later, like their research focus, preferences, or context.

    Args:
        key: Short identifier for the fact (e.g., "research_topic", "preferred_method")
        value: The fact to remember
    """
    MEMORY.add_fact(key, value)
    return f"Remembered: {key} = {value}"


@tool
def recall_fact(key: str) -> str:
    """Recall a previously stored fact.

    Args:
        key: The identifier of the fact to recall
    """
    value = MEMORY.get_fact(key)
    if value:
        return f"{key}: {value}"
    return f"No fact stored for '{key}'"


@tool
def list_known_facts() -> str:
    """List all facts stored in long-term memory."""
    facts = MEMORY.get_all_facts()
    if not facts:
        return "No facts stored yet."
    return "Known facts:\n" + "\n".join([f"- {k}: {v}" for k, v in facts.items()])


@tool
def get_conversation_summary() -> str:
    """Get a summary of the conversation so far."""
    msg_count = len(MEMORY.messages)
    fact_count = len(MEMORY.facts)
    summary = f"Conversation stats:\n- Messages: {msg_count}\n- Facts stored: {fact_count}"
    if MEMORY.summary:
        summary += f"\n\nSummary of earlier discussion:\n{MEMORY.summary}"
    return summary


# ============================================================================
# DOMAIN TOOLS (for demonstration)
# ============================================================================

@tool
def search_literature(query: str) -> str:
    """Search scientific literature for relevant papers.

    Args:
        query: Search query for finding papers
    """
    # Mock results
    return f"""Found 3 relevant papers for "{query}":
1. "Copper-based catalysts for CO2 electroreduction" (2023) - Nature Catalysis
2. "Machine learning for catalyst discovery" (2024) - Science
3. "Density functional theory study of CO2 adsorption" (2023) - JACS"""


@tool
def calculate_property(formula: str, property_name: str) -> str:
    """Calculate a molecular property.

    Args:
        formula: Chemical formula or SMILES
        property_name: Property to calculate (e.g., "molecular_weight", "logP")
    """
    # Mock calculation
    return f"Calculated {property_name} for {formula}: 42.5 (mock value)"


# ============================================================================
# AGENT
# ============================================================================

TOOLS = [
    remember_fact,
    recall_fact,
    list_known_facts,
    get_conversation_summary,
    search_literature,
    calculate_property,
]

SYSTEM_PROMPT = """You are a helpful scientific research assistant with memory capabilities.

You can remember important facts about the user and their research using the remember_fact tool.
Use this proactively when the user shares relevant context like:
- Their research focus or project
- Preferred methods or tools
- Specific compounds or systems they study
- Constraints or requirements

When the user asks about something you've discussed before, use recall_fact to retrieve relevant context.

Be conversational and build on previous discussions. Reference what you remember when relevant."""


def run_conversation():
    """Run interactive conversation with memory."""
    llm = get_llm()

    print(f"\n{'='*60}")
    print("CONVERSATIONAL AGENT WITH MEMORY")
    print(f"{'='*60}")
    print(f"Mode: {LLM_MODE.upper()}")
    print(f"Session: {MEMORY.session_id}")
    print("Type 'quit' to exit, 'facts' to see stored facts")
    print(f"{'='*60}\n")

    if llm is None:
        # Mock mode - simulate conversation
        print("Running in MOCK mode - responses are simulated\n")
        _run_mock_conversation()
        return

    agent = create_agent(llm, TOOLS)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "facts":
            facts = MEMORY.get_all_facts()
            if facts:
                print("\nStored facts:")
                for k, v in facts.items():
                    print(f"  - {k}: {v}")
            else:
                print("\nNo facts stored yet.")
            print()
            continue

        # Add user message to memory
        MEMORY.add_message("user", user_input)

        # Build context with memory
        context = MEMORY.get_context_for_llm()

        # Format for agent
        messages = [("system", SYSTEM_PROMPT)]
        for msg in context:
            if msg["role"] == "system":
                messages.append(("system", msg["content"]))
            elif msg["role"] == "user":
                messages.append(("user", msg["content"]))
            else:
                messages.append(("assistant", msg["content"]))

        # Add current message
        messages.append(("user", user_input))

        try:
            result = agent.invoke({"messages": messages})
            response = result["messages"][-1].content

            # Add response to memory
            MEMORY.add_message("assistant", response)

            print(f"\nAssistant: {response}\n")

        except Exception as e:
            print(f"\nError: {e}\n")


def _run_mock_conversation():
    """Simulated conversation for mock mode."""
    conversations = [
        ("Hi, I'm researching copper catalysts for CO2 reduction.",
         "Hello! That's a fascinating research area. Let me remember that for our conversation.\n[Storing fact: research_topic = copper catalysts for CO2 reduction]\n\nCopper-based catalysts have shown great promise for CO2 electroreduction. Are you focusing on any specific aspects, like selectivity, efficiency, or the reaction mechanism?"),

        ("I'm particularly interested in the selectivity toward ethylene production.",
         "Interesting focus! Ethylene selectivity is one of the key challenges in Cu-catalyzed CO2 reduction.\n[Storing fact: research_focus = ethylene selectivity]\n\nThe C-C coupling step that leads to C2+ products like ethylene is quite sensitive to the catalyst surface structure. Would you like me to search for recent literature on this topic?"),

        ("Yes, please search for recent papers.",
         "Let me search the literature for you.\n[Searching: copper catalyst CO2 reduction ethylene selectivity]\n\nFound 3 relevant papers:\n1. \"Copper-based catalysts for CO2 electroreduction\" (2023) - Nature Catalysis\n2. \"Machine learning for catalyst discovery\" (2024) - Science\n3. \"Density functional theory study of CO2 adsorption\" (2023) - JACS\n\nThe Nature Catalysis paper specifically discusses surface facet effects on C2+ selectivity. Would you like more details on any of these?"),

        ("What do you remember about my research?",
         "Let me recall what I know about your research.\n[Recalling stored facts]\n\nFrom our conversation, I remember:\n- Research topic: copper catalysts for CO2 reduction\n- Research focus: ethylene selectivity\n\nYou're working on understanding and improving the selectivity of copper catalysts toward ethylene production in CO2 electroreduction. This is an important area because achieving high selectivity for C2+ products remains challenging."),
    ]

    for user_msg, assistant_msg in conversations:
        print(f"You: {user_msg}")
        MEMORY.add_message("user", user_msg)

        print(f"\nAssistant: {assistant_msg}\n")
        MEMORY.add_message("assistant", assistant_msg)

        # Extract and store facts from mock responses
        if "research_topic" in assistant_msg:
            MEMORY.add_fact("research_topic", "copper catalysts for CO2 reduction")
        if "research_focus" in assistant_msg:
            MEMORY.add_fact("research_focus", "ethylene selectivity")

    print("\n--- Mock conversation complete ---")
    print("\nStored facts:")
    for k, v in MEMORY.get_all_facts().items():
        print(f"  - {k}: {v}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Conversational Agent with Memory")
    parser.add_argument("--context", "-c", type=str, default=None,
                        help="Initial context to remember")
    parser.add_argument("--load", "-l", type=str, default=None,
                        help="Load memory from file")
    parser.add_argument("--save", "-s", type=str, default=None,
                        help="Save memory to file on exit")
    parser.add_argument("--max-messages", "-m", type=int, default=20,
                        help="Maximum messages to keep in short-term memory")

    args = parser.parse_args()

    # Load existing memory if specified
    if args.load:
        MEMORY = ConversationMemory.load(Path(args.load))
        print(f"Loaded memory from {args.load}")

    MEMORY.max_messages = args.max_messages

    # Add initial context if provided
    if args.context:
        MEMORY.add_fact("initial_context", args.context)

    try:
        run_conversation()
    finally:
        if args.save:
            MEMORY.save(Path(args.save))
            print(f"Memory saved to {args.save}")
