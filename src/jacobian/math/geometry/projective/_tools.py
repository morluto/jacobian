"""Exact rational projective-geometry operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.geometry.projective._arrangements import (
    PROJECTIVE_LINE_ARRANGEMENT_OPERATION,
)
from jacobian.math.geometry.projective._jet import FIRST_JET_OPERATION

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    PROJECTIVE_LINE_ARRANGEMENT_OPERATION,
    FIRST_JET_OPERATION,
)
