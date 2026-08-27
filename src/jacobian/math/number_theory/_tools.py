"""Exact integer number-theory operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.number_theory._additional_ops import ADDITIONAL_NT_OPERATIONS
from jacobian.math.number_theory._derived import DERIVED_NUMBER_THEORY_OPERATIONS
from jacobian.math.number_theory._divisibility import DIVISIBILITY_OPERATIONS
from jacobian.math.number_theory._divisibility_profiles import (
    DIVISIBILITY_PROFILE_OPERATIONS,
)
from jacobian.math.number_theory._finite_abelian_groups import (
    FINITE_ABELIAN_GROUP_FACTORIZATION_OPERATION,
    FINITE_ABELIAN_SPECTRAL_PAIR_OPERATION,
)
from jacobian.math.number_theory._friable import FRIABLE_COUNT_OPERATION
from jacobian.math.number_theory._modular import MODULAR_OPERATIONS
from jacobian.math.number_theory._modular_identity import MODULAR_IDENTITY_OPERATIONS
from jacobian.math.number_theory._periodic import PERIODIC_CONGRUENCE_OPERATIONS
from jacobian.math.number_theory._powerful import POWERFUL_NUMBER_OPERATION
from jacobian.math.number_theory._primes import PRIME_OPERATIONS
from jacobian.math.number_theory._ramanujan_sum import RAMANUJAN_SUM_OPERATION

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    *DIVISIBILITY_OPERATIONS,
    *PRIME_OPERATIONS,
    POWERFUL_NUMBER_OPERATION,
    *MODULAR_OPERATIONS,
    *PERIODIC_CONGRUENCE_OPERATIONS,
    *MODULAR_IDENTITY_OPERATIONS,
    *DERIVED_NUMBER_THEORY_OPERATIONS,
    RAMANUJAN_SUM_OPERATION,
    FINITE_ABELIAN_GROUP_FACTORIZATION_OPERATION,
    FINITE_ABELIAN_SPECTRAL_PAIR_OPERATION,
    FRIABLE_COUNT_OPERATION,
    *ADDITIONAL_NT_OPERATIONS,
    *DIVISIBILITY_PROFILE_OPERATIONS,
)
