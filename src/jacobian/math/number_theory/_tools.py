"""Exact integer number-theory operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.number_theory._derived import DERIVED_NUMBER_THEORY_OPERATIONS
from jacobian.math.number_theory._divisibility import DIVISIBILITY_OPERATIONS
from jacobian.math.number_theory._finite_abelian_groups import (
    FINITE_ABELIAN_GROUP_FACTORIZATION_OPERATION,
)
from jacobian.math.number_theory._friable import FRIABLE_COUNT_OPERATION
from jacobian.math.number_theory._modular import MODULAR_OPERATIONS
from jacobian.math.number_theory._modular_identity import MODULAR_IDENTITY_OPERATIONS
from jacobian.math.number_theory._powerful import POWERFUL_NUMBER_OPERATION
from jacobian.math.number_theory._primes import PRIME_OPERATIONS

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    *DIVISIBILITY_OPERATIONS,
    *PRIME_OPERATIONS,
    POWERFUL_NUMBER_OPERATION,
    *MODULAR_OPERATIONS,
    *MODULAR_IDENTITY_OPERATIONS,
    *DERIVED_NUMBER_THEORY_OPERATIONS,
    FINITE_ABELIAN_GROUP_FACTORIZATION_OPERATION,
    FRIABLE_COUNT_OPERATION,
)
