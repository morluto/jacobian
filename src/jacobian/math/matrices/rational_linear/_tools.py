"""Domain-owned exact rational-linear operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.matrices.rational_linear._operations import (
    rational_linear_operations as _build_tools,
)

__all__ = ["TOOLS"]

TOOLS: MathTools = _build_tools()
