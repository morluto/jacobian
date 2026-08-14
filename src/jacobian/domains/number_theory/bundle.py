"""Installation bundle for exact SymPy-backed number-theory operations."""

from __future__ import annotations

from jacobian.domains.number_theory.checkers import NUMBER_THEORY_EXACT_REPLAY_CHECKERS
from jacobian.domains.number_theory.derived import DERIVED_NUMBER_THEORY_OPERATIONS
from jacobian.domains.number_theory.divisibility import DIVISIBILITY_OPERATIONS
from jacobian.domains.number_theory.finite_abelian_groups import (
    FINITE_ABELIAN_GROUP_FACTORIZATION_OPERATION,
)
from jacobian.domains.number_theory.modular import MODULAR_OPERATIONS
from jacobian.domains.number_theory.modular_identity import (
    MODULAR_IDENTITY_CHECKERS,
    MODULAR_IDENTITY_OPERATIONS,
)
from jacobian.domains.number_theory.primes import PRIME_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def build_number_theory_bundle() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return (
        *DIVISIBILITY_OPERATIONS,
        *PRIME_OPERATIONS,
        *MODULAR_OPERATIONS,
        *MODULAR_IDENTITY_OPERATIONS,
        *DERIVED_NUMBER_THEORY_OPERATIONS,
        FINITE_ABELIAN_GROUP_FACTORIZATION_OPERATION,
    )


CHECKER_DECLARATIONS = (
    *NUMBER_THEORY_EXACT_REPLAY_CHECKERS,
    *MODULAR_IDENTITY_CHECKERS,
)
