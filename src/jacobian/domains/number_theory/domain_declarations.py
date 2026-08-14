"""Exact SymPy-backed number-theory operation declarations."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.number_theory.derived import DERIVED_NUMBER_THEORY_OPERATIONS
from jacobian.domains.number_theory.divisibility import DIVISIBILITY_OPERATIONS
from jacobian.domains.number_theory.finite_abelian_groups import (
    FINITE_ABELIAN_GROUP_FACTORIZATION_OPERATION,
)
from jacobian.domains.number_theory.modular import MODULAR_OPERATIONS
from jacobian.domains.number_theory.modular_identity import MODULAR_IDENTITY_OPERATIONS
from jacobian.domains.number_theory.primes import PRIME_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def number_theory_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        (
            *DIVISIBILITY_OPERATIONS,
            *PRIME_OPERATIONS,
            *MODULAR_OPERATIONS,
            *MODULAR_IDENTITY_OPERATIONS,
            *DERIVED_NUMBER_THEORY_OPERATIONS,
            FINITE_ABELIAN_GROUP_FACTORIZATION_OPERATION,
        ),
        OperationDiagnostic(
            code="INVALID_NUMBER_THEORY_REQUEST",
            stage="number_theory_input_validation",
            message="Input does not satisfy the exact number-theory contract.",
            hint=(
                "Use canonical integer strings and bounded non-negative integers "
                "within each operation's limits."
            ),
        ),
    )
