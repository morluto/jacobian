"""Exact finite-poset operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.posets._closure_tools import CLOSURE_OPERATIONS
from jacobian.math.posets._operations import FINITE_POSET_OPERATIONS

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    *FINITE_POSET_OPERATIONS,
    *CLOSURE_OPERATIONS,
)
