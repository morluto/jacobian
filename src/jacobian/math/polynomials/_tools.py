"""Exact rational polynomial operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.polynomials._cyclotomic_tools import CYCLOTOMIC_OPERATION
from jacobian.math.polynomials._discrete_antiderivative_tools import (
    RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION,
)
from jacobian.math.polynomials._elementary import (
    INTEGER_POLYNOMIAL_OPERATIONS,
)
from jacobian.math.polynomials._elementary_symmetric_tools import (
    ELEMENTARY_SYMMETRIC_FAMILY_OPERATION,
)
from jacobian.math.polynomials._expression_tools import (
    POLYNOMIAL_EXPRESSION_NORMALIZE_OPERATION,
)
from jacobian.math.polynomials._invariants import POLYNOMIAL_INVARIANT_OPERATIONS
from jacobian.math.polynomials._jacobian_syzygy import (
    GRADED_JACOBIAN_SYZYGY_OPERATION,
    JACOBIAN_SYZYGY_COEFFICIENT_LEDGER_OPERATION,
)
from jacobian.math.polynomials._laurent_tools import RATIONAL_LAURENT_MULTIPLY_OPERATION
from jacobian.math.polynomials._mahler_tools import (
    INTEGER_POLYNOMIAL_PROFILE_OPERATIONS,
)
from jacobian.math.polynomials._multiply_ops import POLYNOMIAL_MULTIPLY_OPERATION

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    CYCLOTOMIC_OPERATION,
    RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION,
    *POLYNOMIAL_INVARIANT_OPERATIONS,
    ELEMENTARY_SYMMETRIC_FAMILY_OPERATION,
    POLYNOMIAL_EXPRESSION_NORMALIZE_OPERATION,
    GRADED_JACOBIAN_SYZYGY_OPERATION,
    JACOBIAN_SYZYGY_COEFFICIENT_LEDGER_OPERATION,
    *INTEGER_POLYNOMIAL_OPERATIONS,
    *INTEGER_POLYNOMIAL_PROFILE_OPERATIONS,
    POLYNOMIAL_MULTIPLY_OPERATION,
    RATIONAL_LAURENT_MULTIPLY_OPERATION,
)
