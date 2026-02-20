#!/usr/bin/env python3
"""
Hidden Rule Discovery with LLM Agents

Multiple LLM-powered agents attempt to discover hidden rules governing a simulated world
through experimentation and peer-to-peer communication.

The world has objects with properties (color, size, material, shape) and hidden rules
that determine outcomes of experiments (e.g., "metal objects conduct electricity").

Agents can:
- Propose and run experiments
- Form hypotheses about rules
- Share findings with peers
- Critique and refine hypotheses collaboratively

Configuration via environment variables:
    LLM_PROVIDER: "mock", "anthropic", "openai_compatible" (default: mock)
    LLM_MODEL: Model name (default depends on provider)
    LLM_BASE_URL: Base URL for OpenAI-compatible endpoints
    LLM_API_KEY: API key (or ANTHROPIC_API_KEY for Anthropic)

Example usage:
    # Mock mode (no API calls)
    python hidden_rule_discovery.py

    # Anthropic Claude
    export LLM_PROVIDER=anthropic
    export ANTHROPIC_API_KEY=your-key
    python hidden_rule_discovery.py

    # Argonne inference
    export LLM_PROVIDER=openai_compatible
    export LLM_BASE_URL=https://argonne-inference-endpoint/v1
    export LLM_MODEL=meta-llama/Llama-3-70b-chat-hf
    python hidden_rule_discovery.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from academy.agent import Agent, action, loop
from academy.exchange import LocalExchangeFactory
from academy.handle import Handle
from academy.logging import init_logging
from academy.manager import Manager

from llm_providers import LLMProvider, create_llm_provider

# ----------------------------
# World Definition
# ----------------------------

# Object properties
COLORS = ["red", "blue", "green", "yellow"]
SIZES = ["small", "medium", "large"]
MATERIALS = ["metal", "wood", "glass", "rubber"]
SHAPES = ["sphere", "cube", "pyramid", "cylinder"]

# Experiment types
EXPERIMENTS = [
    "drop {obj} into water",
    "drop {obj} onto the floor",
    "apply electricity to {obj}",
    "expose {obj} to fire",
    "place {obj} in sunlight",
    "put {obj} in the freezer",
    "throw {obj} at a wall",
    "place {obj} on a scale",
]


@dataclass
class Rule:
    """A hidden rule governing the world."""
    condition: str  # e.g., "material == metal"
    experiment_type: str  # e.g., "apply electricity"
    outcome: str  # e.g., "conducts electricity"
    natural_language: str  # e.g., "Metal objects conduct electricity"

    def matches(self, obj: Dict[str, str], experiment: str) -> bool:
        """Check if this rule applies to the given object and experiment."""
        # Check experiment type matches
        if self.experiment_type not in experiment.lower():
            return False

        # Parse and check condition
        return self._check_condition(obj)

    def _check_condition(self, obj: Dict[str, str]) -> bool:
        """Evaluate the condition against object properties."""
        # Simple condition parser: "property == value" or "property != value"
        if "==" in self.condition:
            prop, val = self.condition.split("==")
            return obj.get(prop.strip()) == val.strip()
        elif "!=" in self.condition:
            prop, val = self.condition.split("!=")
            return obj.get(prop.strip()) != val.strip()
        elif self.condition == "always":
            return True
        return False


@dataclass
class HiddenWorld:
    """A world with hidden rules that agents must discover."""
    rules: List[Rule]
    seed: int

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self._experiment_count = 0

    def generate_object(self) -> Dict[str, str]:
        """Generate a random object."""
        return {
            "color": self.rng.choice(COLORS),
            "size": self.rng.choice(SIZES),
            "material": self.rng.choice(MATERIALS),
            "shape": self.rng.choice(SHAPES),
        }

    def describe_object(self, obj: Dict[str, str]) -> str:
        """Generate natural language description of an object."""
        return f"the {obj['size']} {obj['color']} {obj['material']} {obj['shape']}"

    def run_experiment(self, experiment: str) -> str:
        """
        Run an experiment and return the result.

        Experiment should be natural language like:
        "Drop the small red metal sphere into water"
        """
        self._experiment_count += 1

        # Parse the object from the experiment
        obj = self._parse_object_from_experiment(experiment)
        if obj is None:
            return "I don't understand that experiment. Please specify an object with color, size, material, and shape."

        # Find applicable rules and collect outcomes
        outcomes = []
        for rule in self.rules:
            if rule.matches(obj, experiment):
                outcomes.append(rule.outcome)

        # Generate response
        obj_desc = self.describe_object(obj)

        if not outcomes:
            return f"Nothing special happens to {obj_desc}."

        # Combine outcomes
        if len(outcomes) == 1:
            return f"{obj_desc.capitalize()} {outcomes[0]}."
        else:
            return f"{obj_desc.capitalize()} {', and '.join(outcomes)}."

    def _parse_object_from_experiment(self, experiment: str) -> Optional[Dict[str, str]]:
        """Extract object properties from experiment description."""
        exp_lower = experiment.lower()

        obj = {}

        # Find color
        for color in COLORS:
            if color in exp_lower:
                obj["color"] = color
                break

        # Find size
        for size in SIZES:
            if size in exp_lower:
                obj["size"] = size
                break

        # Find material
        for material in MATERIALS:
            if material in exp_lower:
                obj["material"] = material
                break

        # Find shape
        for shape in SHAPES:
            if shape in exp_lower:
                obj["shape"] = shape
                break

        # Default missing properties
        if "color" not in obj:
            obj["color"] = self.rng.choice(COLORS)
        if "size" not in obj:
            obj["size"] = self.rng.choice(SIZES)
        if "material" not in obj:
            obj["material"] = self.rng.choice(MATERIALS)
        if "shape" not in obj:
            obj["shape"] = self.rng.choice(SHAPES)

        return obj

    def get_rules_description(self) -> str:
        """Get natural language description of all rules (for evaluation)."""
        return "\n".join(f"- {rule.natural_language}" for rule in self.rules)

    @property
    def experiment_count(self) -> int:
        return self._experiment_count


def create_world(difficulty: str = "easy", seed: int = 42) -> HiddenWorld:
    """Create a hidden world with rules based on difficulty."""

    if difficulty == "easy":
        # 3 simple, non-conflicting rules
        rules = [
            Rule(
                condition="material == metal",
                experiment_type="electricity",
                outcome="conducts electricity and heats up",
                natural_language="Metal objects conduct electricity"
            ),
            Rule(
                condition="material == glass",
                experiment_type="drop",
                outcome="shatters into pieces",
                natural_language="Glass objects shatter when dropped"
            ),
            Rule(
                condition="material == rubber",
                experiment_type="throw",
                outcome="bounces back",
                natural_language="Rubber objects bounce when thrown"
            ),
        ]

    elif difficulty == "medium":
        # 5 rules with some property interactions
        rules = [
            Rule(
                condition="material == metal",
                experiment_type="electricity",
                outcome="conducts electricity",
                natural_language="Metal objects conduct electricity"
            ),
            Rule(
                condition="material == wood",
                experiment_type="fire",
                outcome="catches fire and burns",
                natural_language="Wood objects burn when exposed to fire"
            ),
            Rule(
                condition="color == blue",
                experiment_type="water",
                outcome="floats",
                natural_language="Blue objects float in water"
            ),
            Rule(
                condition="size == large",
                experiment_type="water",
                outcome="sinks to the bottom",
                natural_language="Large objects sink in water"
            ),
            Rule(
                condition="material == glass",
                experiment_type="freezer",
                outcome="becomes brittle and cracks",
                natural_language="Glass objects crack in the freezer"
            ),
        ]

    elif difficulty == "hard":
        # 8 rules with more complex interactions
        rules = [
            Rule(
                condition="material == metal",
                experiment_type="electricity",
                outcome="conducts electricity",
                natural_language="Metal objects conduct electricity"
            ),
            Rule(
                condition="material == metal",
                experiment_type="sunlight",
                outcome="heats up significantly",
                natural_language="Metal objects heat up in sunlight"
            ),
            Rule(
                condition="material == wood",
                experiment_type="fire",
                outcome="burns",
                natural_language="Wood objects burn in fire"
            ),
            Rule(
                condition="material == wood",
                experiment_type="water",
                outcome="floats",
                natural_language="Wood objects float in water"
            ),
            Rule(
                condition="color == red",
                experiment_type="fire",
                outcome="is immune to flames",
                natural_language="Red objects are fireproof"
            ),
            Rule(
                condition="size == small",
                experiment_type="throw",
                outcome="flies far",
                natural_language="Small objects fly far when thrown"
            ),
            Rule(
                condition="shape == sphere",
                experiment_type="floor",
                outcome="rolls away",
                natural_language="Spheres roll when dropped on the floor"
            ),
            Rule(
                condition="shape == pyramid",
                experiment_type="scale",
                outcome="tips over",
                natural_language="Pyramids tip over on a scale"
            ),
        ]

    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    return HiddenWorld(rules=rules, seed=seed)


# ----------------------------
# Scientist Agent
# ----------------------------

SCIENTIST_SYSTEM_PROMPT = """You are a scientist trying to discover the hidden rules of a mysterious world.

The world contains objects with these properties:
- Colors: red, blue, green, yellow
- Sizes: small, medium, large
- Materials: metal, wood, glass, rubber
- Shapes: sphere, cube, pyramid, cylinder

You can run experiments like:
- "Drop the [size] [color] [material] [shape] into water"
- "Apply electricity to the [size] [color] [material] [shape]"
- "Expose the [size] [color] [material] [shape] to fire"
- "Throw the [size] [color] [material] [shape] at a wall"
- "Place the [size] [color] [material] [shape] in sunlight"
- "Put the [size] [color] [material] [shape] in the freezer"
- "Drop the [size] [color] [material] [shape] onto the floor"

Your goal is to discover the hidden rules that govern how objects behave.
Be systematic - vary one property at a time to isolate what matters.
"""


@dataclass
class Observation:
    """Record of an experiment and its result."""
    experiment: str
    result: str
    step: int


@dataclass
class Hypothesis:
    """A proposed rule about the world."""
    rule: str  # Natural language rule
    confidence: float  # 0-1
    supporting_evidence: List[str]
    contradicting_evidence: List[str]


class ScientistAgent(Agent):
    """An LLM-powered agent that discovers hidden rules through experimentation."""

    def __init__(
        self,
        agent_idx: int,
        world: HiddenWorld,
        llm: LLMProvider,
        comm_prob: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.agent_idx = agent_idx
        self.world = world
        self.llm = llm
        self.comm_prob = comm_prob
        self.rng = random.Random(seed)

        # State
        self.observations: List[Observation] = []
        self.hypotheses: List[Hypothesis] = []
        self.inbox: List[Dict[str, Any]] = []
        self.peers: List[Handle] = []

        # Stats
        self.steps = 0
        self.messages_sent = 0
        self.messages_received = 0
        self.start_time = time.time()

    @action
    async def set_peers(self, peers: List[Handle]) -> None:
        """Set peer handles for communication."""
        self.peers = peers

    @action
    async def receive_message(self, msg: Dict[str, Any]) -> None:
        """Receive a message from a peer."""
        self.messages_received += 1
        if len(self.inbox) > 50:
            self.inbox = self.inbox[-25:]
        self.inbox.append(msg)

    @action
    async def get_state(self) -> Dict[str, Any]:
        """Get current agent state for external observation."""
        return {
            "agent_idx": self.agent_idx,
            "steps": self.steps,
            "n_observations": len(self.observations),
            "n_hypotheses": len(self.hypotheses),
            "hypotheses": [
                {"rule": h.rule, "confidence": h.confidence}
                for h in sorted(self.hypotheses, key=lambda x: -x.confidence)[:5]
            ],
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
        }

    @action
    async def get_hypotheses(self) -> List[Dict[str, Any]]:
        """Get current hypotheses."""
        return [
            {
                "rule": h.rule,
                "confidence": h.confidence,
                "supporting": len(h.supporting_evidence),
                "contradicting": len(h.contradicting_evidence),
            }
            for h in sorted(self.hypotheses, key=lambda x: -x.confidence)
        ]

    def _format_observations(self, last_n: int = 10) -> str:
        """Format recent observations for LLM context."""
        recent = self.observations[-last_n:]
        if not recent:
            return "No experiments run yet."

        lines = []
        for obs in recent:
            lines.append(f"- Experiment: {obs.experiment}")
            lines.append(f"  Result: {obs.result}")
        return "\n".join(lines)

    def _format_hypotheses(self) -> str:
        """Format current hypotheses for LLM context."""
        if not self.hypotheses:
            return "No hypotheses formed yet."

        sorted_hyp = sorted(self.hypotheses, key=lambda x: -x.confidence)[:5]
        lines = []
        for h in sorted_hyp:
            lines.append(f"- {h.rule} (confidence: {h.confidence:.0%})")
        return "\n".join(lines)

    def _format_peer_messages(self) -> str:
        """Format recent peer messages for LLM context."""
        if not self.inbox:
            return "No messages from peers."

        recent = self.inbox[-5:]
        lines = []
        for msg in recent:
            lines.append(f"- Agent {msg.get('from', '?')}: {msg.get('content', '')}")
        return "\n".join(lines)

    async def _propose_experiment(self) -> str:
        """Use LLM to propose next experiment."""
        prompt = f"""Based on your observations and hypotheses, propose the next experiment to run.

Your recent observations:
{self._format_observations()}

Your current hypotheses:
{self._format_hypotheses()}

Messages from peers:
{self._format_peer_messages()}

Propose ONE experiment to run next. Be specific about the object properties.
Format: Just write the experiment, e.g., "Drop the small red metal sphere into water"
"""

        response = self.llm.complete(SCIENTIST_SYSTEM_PROMPT, prompt, max_tokens=100)

        # Extract experiment from response
        experiment = response.strip().strip('"').strip("'")

        # Ensure it's a valid experiment format
        if not any(exp_type in experiment.lower() for exp_type in
                   ["drop", "electricity", "fire", "throw", "sunlight", "freezer", "floor", "scale"]):
            # Default to a random experiment
            obj = self.world.generate_object()
            exp_template = self.rng.choice(EXPERIMENTS)
            experiment = exp_template.format(obj=self.world.describe_object(obj))

        return experiment

    async def _update_hypotheses(self) -> None:
        """Use LLM to update hypotheses based on observations."""
        if len(self.observations) < 3:
            return

        prompt = f"""Based on all your observations, update your hypotheses about the hidden rules.

Your observations:
{self._format_observations(last_n=20)}

Your current hypotheses:
{self._format_hypotheses()}

Messages from peers:
{self._format_peer_messages()}

Analyze the patterns and propose updated hypotheses. For each hypothesis:
1. State the rule clearly
2. Rate your confidence (0-100%)

Format each hypothesis as:
RULE: [the rule]
CONFIDENCE: [0-100]

Propose up to 5 hypotheses, focusing on ones with strong evidence.
"""

        response = self.llm.complete(SCIENTIST_SYSTEM_PROMPT, prompt, max_tokens=500)

        # Parse hypotheses from response
        new_hypotheses = []

        # Simple parsing - look for RULE: and CONFIDENCE: patterns
        rule_pattern = r'RULE:\s*(.+?)(?=CONFIDENCE:|RULE:|$)'
        conf_pattern = r'CONFIDENCE:\s*(\d+)'

        rules = re.findall(rule_pattern, response, re.IGNORECASE | re.DOTALL)
        confs = re.findall(conf_pattern, response, re.IGNORECASE)

        for i, rule in enumerate(rules):
            rule = rule.strip()
            if not rule:
                continue

            conf = int(confs[i]) / 100 if i < len(confs) else 0.5
            conf = max(0, min(1, conf))

            new_hypotheses.append(Hypothesis(
                rule=rule,
                confidence=conf,
                supporting_evidence=[],
                contradicting_evidence=[],
            ))

        if new_hypotheses:
            self.hypotheses = new_hypotheses

    async def _maybe_share(self) -> None:
        """Maybe share findings with a peer."""
        if not self.peers or self.rng.random() > self.comm_prob:
            return

        if not self.hypotheses and not self.observations:
            return

        peer = self.rng.choice(self.peers)

        # Compose message
        top_hypotheses = sorted(self.hypotheses, key=lambda x: -x.confidence)[:3]
        content_parts = []

        if top_hypotheses:
            hyp_strs = [f"{h.rule} ({h.confidence:.0%} confident)" for h in top_hypotheses]
            content_parts.append(f"My top hypotheses: {'; '.join(hyp_strs)}")

        if self.observations:
            recent = self.observations[-3:]
            obs_strs = [f"'{o.experiment}' -> '{o.result}'" for o in recent]
            content_parts.append(f"Recent findings: {'; '.join(obs_strs)}")

        msg = {
            "from": self.agent_idx,
            "content": " | ".join(content_parts),
            "hypotheses": [{"rule": h.rule, "confidence": h.confidence} for h in top_hypotheses],
            "step": self.steps,
        }

        await peer.receive_message(msg)
        self.messages_sent += 1

    @loop
    async def run(self, shutdown: asyncio.Event) -> None:
        """Main agent loop."""

        while not shutdown.is_set():
            self.steps += 1

            # 1. Propose and run experiment
            experiment = await self._propose_experiment()
            result = self.world.run_experiment(experiment)

            self.observations.append(Observation(
                experiment=experiment,
                result=result,
                step=self.steps,
            ))

            # 2. Update hypotheses periodically
            if self.steps % 5 == 0:
                await self._update_hypotheses()

            # 3. Maybe share with peers
            await self._maybe_share()

            # 4. Clear old inbox messages
            if len(self.inbox) > 20:
                self.inbox = self.inbox[-10:]

            # 5. Pace the loop
            await asyncio.sleep(0.5)  # Slower than polynomial fitting due to LLM calls


# ----------------------------
# Evaluation
# ----------------------------

def evaluate_hypotheses(
    agent_hypotheses: List[Dict[str, Any]],
    true_rules: List[Rule],
    llm: LLMProvider,
) -> Dict[str, Any]:
    """
    Evaluate how well agent hypotheses match the true rules.
    Uses LLM to assess semantic similarity.
    """

    true_rules_str = "\n".join(f"- {r.natural_language}" for r in true_rules)
    agent_hyp_str = "\n".join(
        f"- {h['rule']} (confidence: {h['confidence']:.0%})"
        for h in agent_hypotheses[:10]
    )

    prompt = f"""Compare these discovered hypotheses against the true hidden rules.

TRUE HIDDEN RULES:
{true_rules_str}

DISCOVERED HYPOTHESES:
{agent_hyp_str}

For each true rule, assess if it was discovered (fully, partially, or not at all).
Then provide an overall score from 0-100.

Format:
RULE_SCORES:
- [true rule 1]: [fully/partially/missed]
- [true rule 2]: [fully/partially/missed]
...

OVERALL_SCORE: [0-100]
EXPLANATION: [brief explanation]
"""

    response = llm.complete(
        "You are an evaluator assessing scientific discovery accuracy.",
        prompt,
        max_tokens=500
    )

    # Parse score
    score_match = re.search(r'OVERALL_SCORE:\s*(\d+)', response)
    score = int(score_match.group(1)) if score_match else 50

    return {
        "score": score,
        "evaluation": response,
        "n_true_rules": len(true_rules),
        "n_hypotheses": len(agent_hypotheses),
    }


# ----------------------------
# Main
# ----------------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hidden Rule Discovery with LLM Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  LLM_PROVIDER    Provider: mock, anthropic, openai_compatible (default: mock)
  LLM_MODEL       Model name (provider-specific)
  LLM_BASE_URL    Base URL for OpenAI-compatible endpoints
  LLM_API_KEY     API key (or ANTHROPIC_API_KEY for Anthropic)

Examples:
  # Mock mode (no API calls, for testing)
  python hidden_rule_discovery.py

  # With Anthropic Claude
  LLM_PROVIDER=anthropic python hidden_rule_discovery.py

  # With Argonne inference
  LLM_PROVIDER=openai_compatible LLM_BASE_URL=https://... python hidden_rule_discovery.py
        """
    )

    parser.add_argument(
        "--provider", "-p",
        type=str,
        default=None,
        help="LLM provider: mock, anthropic, openai_compatible (default: from LLM_PROVIDER or mock)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name (default: provider-specific)"
    )
    parser.add_argument(
        "--agents", "-a",
        type=int,
        default=4,
        help="Number of agents (default: 4)"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=50,
        help="Number of steps to run (default: 50)"
    )
    parser.add_argument(
        "--difficulty", "-d",
        type=str,
        choices=["easy", "medium", "hard"],
        default="medium",
        help="Difficulty level (default: medium)"
    )
    parser.add_argument(
        "--comm-prob", "-c",
        type=float,
        default=0.3,
        help="Communication probability (default: 0.3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--no-comm",
        action="store_true",
        help="Disable communication between agents"
    )

    return parser.parse_args()


async def main():
    args = parse_args()
    init_logging("INFO")

    # Configuration
    N_AGENTS = args.agents
    DIFFICULTY = args.difficulty
    STEPS = args.steps
    COMM_PROB = 0.0 if args.no_comm else args.comm_prob
    SEED = args.seed

    print("=" * 70)
    print("Hidden Rule Discovery with LLM Agents")
    print("=" * 70)

    # Create LLM provider
    llm = create_llm_provider(
        provider=args.provider,
        model=args.model,
        seed=SEED,
    )
    print(f"\nLLM Provider: {llm.name}")

    # Create world
    world = create_world(difficulty=DIFFICULTY, seed=SEED)
    print(f"Difficulty: {DIFFICULTY}")
    print(f"Number of hidden rules: {len(world.rules)}")
    print(f"Number of agents: {N_AGENTS}")
    print(f"Communication probability: {COMM_PROB}")

    print("\n[Hidden rules - for reference only]")
    print(world.get_rules_description())
    print()

    # Launch agents
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(max_workers=N_AGENTS + 4),
    ) as manager:

        handles: List[Handle[ScientistAgent]] = []
        for i in range(N_AGENTS):
            h = await manager.launch(
                ScientistAgent,
                args=(i, world, llm, COMM_PROB, SEED + i),
            )
            handles.append(h)

        # Wire full connectivity (everyone can talk to everyone)
        for i, h in enumerate(handles):
            peers = [handles[j] for j in range(N_AGENTS) if j != i]
            await h.set_peers(peers)

        print(f"Launched {N_AGENTS} agents, running for {STEPS} steps...")
        print()

        # Run and observe
        for step in range(1, STEPS + 1):
            await asyncio.sleep(1.0)  # Let agents run

            if step % 10 == 0:
                print(f"\n--- Step {step} ---")

                states = await asyncio.gather(*[h.get_state() for h in handles])

                for state in states:
                    print(f"\nAgent {state['agent_idx']}:")
                    print(f"  Observations: {state['n_observations']}")
                    print(f"  Messages: sent={state['messages_sent']}, recv={state['messages_received']}")
                    if state['hypotheses']:
                        print(f"  Top hypotheses:")
                        for hyp in state['hypotheses'][:3]:
                            print(f"    - {hyp['rule'][:60]}... ({hyp['confidence']:.0%})")

        # Final evaluation
        print("\n" + "=" * 70)
        print("FINAL EVALUATION")
        print("=" * 70)

        all_hypotheses = await asyncio.gather(*[h.get_hypotheses() for h in handles])

        for i, hypotheses in enumerate(all_hypotheses):
            print(f"\nAgent {i} final hypotheses:")
            for hyp in hypotheses[:5]:
                print(f"  - {hyp['rule'][:70]}... ({hyp['confidence']:.0%})")

        # Evaluate best agent
        print("\n--- Evaluation ---")

        # Combine all hypotheses and deduplicate by similarity
        combined = []
        for hyps in all_hypotheses:
            combined.extend(hyps)

        # Sort by confidence
        combined = sorted(combined, key=lambda x: -x["confidence"])[:10]

        eval_result = evaluate_hypotheses(combined, world.rules, llm)
        print(f"\nOverall discovery score: {eval_result['score']}/100")
        print(f"\nEvaluation details:\n{eval_result['evaluation']}")

        print(f"\nTotal experiments run: {world.experiment_count}")

        # Shutdown
        for h in handles:
            await h.shutdown()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
