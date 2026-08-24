"""Validated real-analysis operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.analysis._operations import (
    BOX_EXPRESSION_ENCLOSURE_OPERATIONS,
    EXPRESSION_ENCLOSURE_OPERATIONS,
    POINT_ENCLOSURE_OPERATIONS,
    SECOND_JET_ENCLOSURE_OPERATIONS,
)

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    *POINT_ENCLOSURE_OPERATIONS,
    *EXPRESSION_ENCLOSURE_OPERATIONS,
    *BOX_EXPRESSION_ENCLOSURE_OPERATIONS,
    *SECOND_JET_ENCLOSURE_OPERATIONS,
)
