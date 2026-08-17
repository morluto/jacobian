"""Finite simplicial topology domain."""

from jacobian.catalog.models import MathTools
from jacobian.math.topology._operations import TOPOLOGY_OPERATIONS

__all__ = ["TOOLS"]

TOOLS: MathTools = TOPOLOGY_OPERATIONS
