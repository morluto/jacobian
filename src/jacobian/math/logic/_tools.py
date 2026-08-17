"""Exact logic operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.logic._operations import LOGIC_OPERATIONS

__all__ = ["TOOLS"]

TOOLS: MathTools = LOGIC_OPERATIONS
