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
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Global logging flag - set via command line
AGENT_LOGGING_ENABLED = False

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

# Experiment type keywords for parsing
EXPERIMENT_KEYWORDS = {
    "water": "drop_water",
    "floor": "drop_floor",
    "electricity": "electricity",
    "fire": "fire",
    "sunlight": "sunlight",
    "freezer": "freezer",
    "throw": "throw",
    "wall": "throw",
    "scale": "scale",
}

# Outcome keywords for parsing
OUTCOME_KEYWORDS = [
    "conducts", "heats", "shatters", "breaks", "cracks", "burns", "catches fire",
    "floats", "sinks", "bounces", "rolls", "tips", "flies", "nothing", "immune"
]


# ----------------------------
# Structured Hypothesis Mining
# ----------------------------

@dataclass
class StructuredObservation:
    """Parsed observation in structured form."""
    color: str
    size: str
    material: str
    shape: str
    experiment_type: str
    outcome: str  # "nothing" or the key outcome word
    raw_experiment: str
    raw_result: str


def parse_observation(experiment: str, result: str) -> Optional[StructuredObservation]:
    """Parse experiment and result into structured form."""
    exp_lower = experiment.lower()
    res_lower = result.lower()

    # Extract object properties
    color = next((c for c in COLORS if c in exp_lower), None)
    size = next((s for s in SIZES if s in exp_lower), None)
    material = next((m for m in MATERIALS if m in exp_lower), None)
    shape = next((sh for sh in SHAPES if sh in exp_lower), None)

    # Extract experiment type
    exp_type = None
    for keyword, etype in EXPERIMENT_KEYWORDS.items():
        if keyword in exp_lower:
            exp_type = etype
            break

    # Extract outcome
    if "nothing" in res_lower:
        outcome = "nothing"
    else:
        outcome = next((kw for kw in OUTCOME_KEYWORDS if kw in res_lower), "unknown")

    if not all([color, size, material, shape, exp_type]):
        return None

    return StructuredObservation(
        color=color,
        size=size,
        material=material,
        shape=shape,
        experiment_type=exp_type,
        outcome=outcome,
        raw_experiment=experiment,
        raw_result=result,
    )


class HypothesisMiner:
    """Mines hypotheses from structured observations using correlation analysis."""

    def __init__(self):
        # Track observations: (property_type, property_value, experiment_type) -> {outcome: count}
        self.correlations: Dict[Tuple[str, str, str], Dict[str, int]] = {}
        self.total_by_key: Dict[Tuple[str, str, str], int] = {}
        self.observations: List[StructuredObservation] = []

    def add_observation(self, obs: StructuredObservation) -> None:
        """Add an observation and update correlations."""
        self.observations.append(obs)

        # Track correlations for each property type
        for prop_type, prop_value in [
            ("color", obs.color),
            ("size", obs.size),
            ("material", obs.material),
            ("shape", obs.shape),
        ]:
            key = (prop_type, prop_value, obs.experiment_type)
            if key not in self.correlations:
                self.correlations[key] = {}
                self.total_by_key[key] = 0

            self.total_by_key[key] += 1
            self.correlations[key][obs.outcome] = self.correlations[key].get(obs.outcome, 0) + 1

    def get_hypotheses(self, min_evidence: int = 2, min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """Generate hypotheses from observed correlations."""
        hypotheses = []

        for key, outcomes in self.correlations.items():
            prop_type, prop_value, exp_type = key
            total = self.total_by_key[key]

            if total < min_evidence:
                continue

            for outcome, count in outcomes.items():
                if outcome == "nothing" or outcome == "unknown":
                    continue

                confidence = count / total
                if confidence < min_confidence:
                    continue

                # Generate natural language rule
                exp_readable = exp_type.replace("_", " ").replace("drop water", "dropped in water").replace("drop floor", "dropped")
                rule_text = f"{prop_value.capitalize()} objects {outcome} when {exp_readable}"

                hypotheses.append({
                    "rule": rule_text,
                    "condition": {prop_type: prop_value},
                    "experiment_type": exp_type,
                    "outcome": outcome,
                    "confidence": confidence,
                    "evidence_count": count,
                    "total_observations": total,
                })

        # Sort by evidence count, then confidence
        hypotheses.sort(key=lambda h: (-h["evidence_count"], -h["confidence"]))

        # Deduplicate: prefer simpler hypotheses (same outcome, more general)
        return self._simplify_hypotheses(hypotheses)

    def _simplify_hypotheses(self, hypotheses: List[Dict]) -> List[Dict]:
        """Remove redundant hypotheses, keeping the simplest."""
        # Group by outcome
        by_outcome: Dict[str, List[Dict]] = {}
        for h in hypotheses:
            outcome = h["outcome"]
            if outcome not in by_outcome:
                by_outcome[outcome] = []
            by_outcome[outcome].append(h)

        simplified = []
        for outcome, hyps in by_outcome.items():
            # For each outcome, keep the hypothesis with most evidence
            # If material-based has good evidence, prefer it over color/size/shape
            material_hyps = [h for h in hyps if "material" in h["condition"]]
            other_hyps = [h for h in hyps if "material" not in h["condition"]]

            if material_hyps and material_hyps[0]["confidence"] >= 0.7:
                simplified.append(material_hyps[0])
            elif hyps:
                simplified.append(hyps[0])

        return simplified

    def merge_from(self, other: 'HypothesisMiner') -> None:
        """Merge observations from another miner (for peer sharing)."""
        for obs in other.observations:
            if obs not in self.observations:
                self.add_observation(obs)


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

HYPOTHESIS_SYSTEM_PROMPT = """You are a scientist analyzing experimental data to discover hidden rules.
Your ONLY task is to output hypotheses in this exact format:

RULE: [the rule you discovered]
CONFIDENCE: [0-100]

Do NOT suggest experiments. ONLY output RULE/CONFIDENCE pairs."""

SCIENTIST_SYSTEM_PROMPT = """You propose experiments to discover hidden rules about objects.
Format: "[action] the [size] [color] [material] [shape]"
Example: "Drop the small red glass sphere into water"
Be systematic - explore different materials and experiment types."""


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

        # Structured hypothesis mining
        self.miner = HypothesisMiner()

        # Stats
        self.steps = 0
        self.messages_sent = 0
        self.messages_received = 0
        self.start_time = time.time()

        # Logging counters
        self.llm_query_count = 0
        self.msg_log_count = 0

    def _log(self, msg_type: str, content: str) -> None:
        """Log agent activity if logging is enabled."""
        if not AGENT_LOGGING_ENABLED:
            return

        prefix = f"[Agent {self.agent_idx}]"
        # Print to stderr to avoid mixing with regular output
        print(f"{prefix} {msg_type}: {content}", file=sys.stderr, flush=True)

    @action
    async def set_peers(self, peers: List[Handle]) -> None:
        """Set peer handles for communication."""
        self.peers = peers

    @action
    async def receive_message(self, msg: Dict[str, Any]) -> None:
        """Receive a message from a peer."""
        self.messages_received += 1
        self.msg_log_count += 1
        self._log(f"MSG_RECV #{self.msg_log_count}", f"from=Agent {msg.get('from', '?')} | {msg.get('content', '')[:200]}")

        # Merge structured observations from peer
        structured_obs = msg.get("structured_observations", [])
        new_obs_count = 0
        for obs_dict in structured_obs:
            obs = StructuredObservation(
                color=obs_dict["color"],
                size=obs_dict["size"],
                material=obs_dict["material"],
                shape=obs_dict["shape"],
                experiment_type=obs_dict["experiment_type"],
                outcome=obs_dict["outcome"],
                raw_experiment="",
                raw_result="",
            )
            # Only add if we don't have this exact observation
            if obs not in self.miner.observations:
                self.miner.add_observation(obs)
                new_obs_count += 1

        if new_obs_count > 0:
            self._log(f"MERGED_OBS", f"added {new_obs_count} observations from Agent {msg.get('from', '?')}")

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
        # Count experiments by type to encourage diversity
        exp_counts: Dict[str, int] = {}
        for obs in self.miner.observations:
            exp_counts[obs.experiment_type] = exp_counts.get(obs.experiment_type, 0) + 1

        # Find least-tested experiment types
        all_exp_types = ["drop_water", "drop_floor", "electricity", "fire", "throw", "sunlight", "freezer"]
        sorted_by_count = sorted(all_exp_types, key=lambda e: exp_counts.get(e, 0))
        least_tested = sorted_by_count[:3]

        # Map to readable names
        exp_names = {
            "drop_water": "drop into water",
            "drop_floor": "drop onto floor",
            "electricity": "apply electricity",
            "fire": "expose to fire",
            "throw": "throw at wall",
            "sunlight": "place in sunlight",
            "freezer": "put in freezer",
        }
        suggested = [exp_names.get(e, e) for e in least_tested]

        prompt = f"""Propose ONE experiment to test how objects behave.

MATERIALS: metal, wood, glass, rubber
EXPERIMENTS: drop into water, drop onto floor, apply electricity, expose to fire, throw at wall

IMPORTANT: We need MORE of these experiment types: {', '.join(suggested)}

Format: "[action] the [size] [color] [material] [shape]"
Example: "Apply electricity to the small red metal cube"

Your experiment:"""

        self.llm_query_count += 1
        query_num = self.llm_query_count
        self._log(f"LLM_QUERY #{query_num}", f"propose_experiment:\n{prompt}")

        response = self.llm.complete(SCIENTIST_SYSTEM_PROMPT, prompt, max_tokens=100)

        self._log(f"LLM_RESPONSE #{query_num}", f"{response.strip()}")

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

        prompt = f"""Observations from experiments:
{self._format_observations(last_n=20)}

Messages from peers:
{self._format_peer_messages()}

Based on these observations, here are the hypotheses about hidden rules:

RULE: """

        self.llm_query_count += 1
        query_num = self.llm_query_count
        self._log(f"LLM_QUERY #{query_num}", f"update_hypotheses:\n{prompt}")

        response = self.llm.complete(HYPOTHESIS_SYSTEM_PROMPT, prompt, max_tokens=500)

        self._log(f"LLM_RESPONSE #{query_num}", f"{response.strip()}")

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
            self._log(f"HYPOTHESES_UPDATED", f"parsed {len(new_hypotheses)} hypotheses")
        else:
            self._log(f"HYPOTHESES_PARSE_FAILED", f"no RULE:/CONFIDENCE: patterns found. Response was:\n{response}")

    async def _simplify_hypotheses(self) -> None:
        """Ask LLM to generalize overly specific hypotheses using Occam's razor."""
        if not self.hypotheses:
            return

        hyp_list = "\n".join(f"- {h.rule}" for h in self.hypotheses[:5])

        prompt = f"""Simplify these hypotheses to their CORE pattern. Remove ALL unnecessary specifics.

Current hypotheses:
{hyp_list}

For each hypothesis, ask: "What is the MINIMUM rule that explains this?"

ALWAYS simplify:
- "Glass breaks when dropped IN WATER" -> "Glass breaks when dropped" (location irrelevant)
- "Glass SPHERES shatter" -> "Glass shatters" (shape probably irrelevant)
- "Red metal conducts" -> "Metal conducts" (color irrelevant)
- "Small rubber bounces when thrown AT WALLS" -> "Rubber bounces" (target irrelevant)

Focus on: MATERIAL, and the ACTION. Remove colors, sizes, shapes, locations unless truly essential.

Output the simplified core rules:

RULE: """

        self.llm_query_count += 1
        query_num = self.llm_query_count
        self._log(f"LLM_QUERY #{query_num}", f"simplify_hypotheses:\n{prompt}")

        response = self.llm.complete(HYPOTHESIS_SYSTEM_PROMPT, prompt, max_tokens=500)

        self._log(f"LLM_RESPONSE #{query_num}", f"{response.strip()}")

        # Parse simplified hypotheses
        new_hypotheses = []
        rule_pattern = r'RULE:\s*(.+?)(?=CONFIDENCE:|RULE:|$)'
        conf_pattern = r'CONFIDENCE:\s*(\d+)'

        # Prepend "RULE: " since prompt ends with it
        full_response = "RULE: " + response

        rules = re.findall(rule_pattern, full_response, re.IGNORECASE | re.DOTALL)
        confs = re.findall(conf_pattern, full_response, re.IGNORECASE)

        for i, rule in enumerate(rules):
            rule = rule.strip()
            if not rule:
                continue

            conf = int(confs[i]) / 100 if i < len(confs) else 0.7
            conf = max(0, min(1, conf))

            new_hypotheses.append(Hypothesis(
                rule=rule,
                confidence=conf,
                supporting_evidence=[],
                contradicting_evidence=[],
            ))

        if new_hypotheses:
            self.hypotheses = new_hypotheses
            self._log(f"HYPOTHESES_SIMPLIFIED", f"simplified to {len(new_hypotheses)} hypotheses")

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

        # Include recent structured observations for peer to merge
        recent_structured = self.miner.observations[-5:] if self.miner.observations else []

        msg = {
            "from": self.agent_idx,
            "content": " | ".join(content_parts),
            "hypotheses": [{"rule": h.rule, "confidence": h.confidence} for h in top_hypotheses],
            "structured_observations": [
                {"color": o.color, "size": o.size, "material": o.material, "shape": o.shape,
                 "experiment_type": o.experiment_type, "outcome": o.outcome}
                for o in recent_structured
            ],
            "step": self.steps,
        }

        try:
            await peer.receive_message(msg)
            self.messages_sent += 1
            self.msg_log_count += 1
            self._log(f"MSG_SEND #{self.msg_log_count}", f"to=peer | {msg['content'][:200]}")
        except Exception:
            # Peer may have terminated during shutdown - ignore
            pass

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

            # 2. Parse observation and add to miner
            structured_obs = parse_observation(experiment, result)
            if structured_obs:
                self.miner.add_observation(structured_obs)
                self._log("OBSERVATION", f"{structured_obs.material}/{structured_obs.experiment_type} -> {structured_obs.outcome}")

            # 3. Update hypotheses from miner periodically
            if self.steps % 5 == 0:
                mined_hypotheses = self.miner.get_hypotheses(min_evidence=2, min_confidence=0.6)
                self.hypotheses = [
                    Hypothesis(
                        rule=h["rule"],
                        confidence=h["confidence"],
                        supporting_evidence=[],
                        contradicting_evidence=[],
                    )
                    for h in mined_hypotheses[:5]
                ]
                if mined_hypotheses:
                    self._log("HYPOTHESES_MINED", f"{len(mined_hypotheses)} rules: {[h['rule'] for h in mined_hypotheses[:3]]}")

            # 4. Maybe share with peers
            await self._maybe_share()

            # 5. Clear old inbox messages
            if len(self.inbox) > 20:
                self.inbox = self.inbox[-10:]

            # 6. Pace the loop
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
    Uses LLM to assess semantic similarity for each hypothesis.
    """

    results = {
        "hypotheses_evaluation": [],  # Each hypothesis with TRUE/FALSE
        "rules_found": [],            # True rules that were discovered
        "rules_missed": [],           # True rules that were NOT discovered
        "score": 0,
        "n_true_rules": len(true_rules),
        "n_hypotheses": len(agent_hypotheses),
    }

    true_rules_str = "\n".join(f"- {r.natural_language}" for r in true_rules)
    rules_matched = set()

    # Evaluate each hypothesis
    for hyp in agent_hypotheses[:10]:
        prompt = f"""TRUE RULES:
{true_rules_str}

HYPOTHESIS: {hyp['rule']}

Does this hypothesis capture the essence of any true rule? A hypothesis matches if it identifies the same core relationship, even if it's more specific or uses different wording.

Examples of matches:
- "Glass breaks when dropped in water" matches "Glass objects shatter when dropped"
- "Metal conducts electricity when touched" matches "Metal objects conduct electricity"

Reply with EXACTLY one of:
MATCH: [rule number 1-{len(true_rules)}]
NO_MATCH

Answer: """

        response = llm.complete(
            "You are evaluating if a hypothesis captures the essence of any true rule. Be lenient - partial matches count.",
            prompt,
            max_tokens=50
        )

        is_match = False
        matched_rule = None

        if "MATCH" in response.upper() and "NO_MATCH" not in response.upper():
            # Try to extract which rule matched
            match = re.search(r'MATCH[:\s]*(\d+)', response, re.IGNORECASE)
            if match:
                rule_idx = int(match.group(1)) - 1
                if 0 <= rule_idx < len(true_rules):
                    is_match = True
                    matched_rule = true_rules[rule_idx].natural_language
                    rules_matched.add(rule_idx)
            else:
                # Generic match without rule number
                is_match = True

        results["hypotheses_evaluation"].append({
            "hypothesis": hyp['rule'],
            "confidence": hyp['confidence'],
            "verdict": "TRUE" if is_match else "FALSE",
            "matched_rule": matched_rule,
        })

    # Determine which rules were found vs missed
    for i, rule in enumerate(true_rules):
        if i in rules_matched:
            results["rules_found"].append(rule.natural_language)
        else:
            results["rules_missed"].append(rule.natural_language)

    # Calculate score: percentage of true rules discovered
    results["score"] = int(100 * len(results["rules_found"]) / len(true_rules)) if true_rules else 0

    return results


def format_evaluation_summary(eval_result: Dict[str, Any]) -> str:
    """Format the evaluation results for display."""
    lines = []

    lines.append("DISCOVERED HYPOTHESES:")
    for item in eval_result["hypotheses_evaluation"]:
        verdict = item["verdict"]
        hyp = item["hypothesis"][:60]
        conf = item["confidence"]
        if item["matched_rule"]:
            lines.append(f"  [{verdict}] {hyp}... ({conf:.0%}) -> matches: {item['matched_rule']}")
        else:
            lines.append(f"  [{verdict}] {hyp}... ({conf:.0%})")

    lines.append("")
    lines.append(f"RULES FOUND ({len(eval_result['rules_found'])}/{eval_result['n_true_rules']}):")
    for rule in eval_result["rules_found"]:
        lines.append(f"  + {rule}")

    if eval_result["rules_missed"]:
        lines.append("")
        lines.append(f"RULES MISSED ({len(eval_result['rules_missed'])}):")
        for rule in eval_result["rules_missed"]:
            lines.append(f"  - {rule}")

    lines.append("")
    lines.append(f"SCORE: {eval_result['score']}/100")

    return "\n".join(lines)


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
    parser.add_argument(
        "--log-agents",
        action="store_true",
        help="Enable verbose logging of agent LLM queries, responses, and messages"
    )

    return parser.parse_args()


async def main():
    global AGENT_LOGGING_ENABLED

    args = parse_args()
    init_logging("WARNING")

    # Enable agent logging if requested
    if args.log_agents:
        AGENT_LOGGING_ENABLED = True
        print("Agent logging enabled (output to stderr)", file=sys.stderr)

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
        print(f"\n{format_evaluation_summary(eval_result)}")

        print(f"\nTotal experiments run: {world.experiment_count}")

        # Shutdown
        for h in handles:
            await h.shutdown()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
