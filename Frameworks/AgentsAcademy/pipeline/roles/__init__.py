"""Role-based agents for the scientific discovery pipeline."""

from .scout import ScoutAgent
from .planner import PlannerAgent
from .operator import OperatorAgent
from .analyst import AnalystAgent
from .archivist import ArchivistAgent

__all__ = [
    "ScoutAgent",
    "PlannerAgent",
    "OperatorAgent",
    "AnalystAgent",
    "ArchivistAgent",
]
