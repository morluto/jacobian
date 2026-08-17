"""Exact rational polynomial operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.polynomials._elementary import (
    INTEGER_POLYNOMIAL_OPERATIONS,
    RATIONAL_POLYNOMIAL_OPERATIONS,
)
from jacobian.math.polynomials._invariants import POLYNOMIAL_INVARIANT_OPERATIONS
from jacobian.math.polynomials._jacobian_syzygy import (
    GRADED_JACOBIAN_SYZYGY_OPERATION,
    JACOBIAN_SYZYGY_COEFFICIENT_LEDGER_OPERATION,
)

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    *POLYNOMIAL_INVARIANT_OPERATIONS,
    GRADED_JACOBIAN_SYZYGY_OPERATION,
    JACOBIAN_SYZYGY_COEFFICIENT_LEDGER_OPERATION,
    *INTEGER_POLYNOMIAL_OPERATIONS,
    *RATIONAL_POLYNOMIAL_OPERATIONS,
)
