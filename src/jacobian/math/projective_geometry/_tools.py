"""Exact rational projective-geometry operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.projective_geometry._arrangements import (
    PROJECTIVE_LINE_ARRANGEMENT_OPERATION,
)

__all__ = ["TOOLS"]

TOOLS: MathTools = (PROJECTIVE_LINE_ARRANGEMENT_OPERATION,)
