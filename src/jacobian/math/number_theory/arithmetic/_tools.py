"""Exact integer and rational arithmetic operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.number_theory.arithmetic._integers import INTEGER_OPERATIONS
from jacobian.math.number_theory.arithmetic._multiplicative_tools import (
    MULTIPLICATIVE_FORM_OPERATIONS,
)
from jacobian.math.number_theory.arithmetic._rationals import RATIONAL_OPERATIONS
from jacobian.math.number_theory.arithmetic._real_quadratic import (
    REAL_QUADRATIC_OPERATIONS,
)

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    *INTEGER_OPERATIONS,
    *RATIONAL_OPERATIONS,
    *REAL_QUADRATIC_OPERATIONS,
    *MULTIPLICATIVE_FORM_OPERATIONS,
)
