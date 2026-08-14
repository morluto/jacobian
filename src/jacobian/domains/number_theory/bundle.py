"""Installation bundle for exact SymPy-backed number-theory operations."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domain_bundles import DomainBundle
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
from jacobian.operations import (
    DomainDiagnostics,
    DomainSemantics,
)


def build_number_theory_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="number_theory",
        schema_namespace="jacobian.number-theory",
        semantics=DomainSemantics(
            name="jacobian.exact-integer-number-theory",
            version="2",
            definition={
                "description": (
                    "Exact integer divisibility, primes, arithmetic functions, "
                    "modular arithmetic, and bounded finite abelian group "
                    "factorization"
                ),
                "integer_encoding": "canonical decimal string",
            },
        ),
        operations=(
            *DIVISIBILITY_OPERATIONS,
            *PRIME_OPERATIONS,
            *MODULAR_OPERATIONS,
            *MODULAR_IDENTITY_OPERATIONS,
            *DERIVED_NUMBER_THEORY_OPERATIONS,
            FINITE_ABELIAN_GROUP_FACTORIZATION_OPERATION,
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=OperationDiagnostic(
                code="INVALID_NUMBER_THEORY_REQUEST",
                stage="number_theory_input_validation",
                message="Input does not satisfy the exact number-theory contract.",
                hint=(
                    "Use canonical integer strings and bounded non-negative "
                    "integers within each operation's limits."
                ),
            )
        ),
        checker_declarations=(
            *NUMBER_THEORY_EXACT_REPLAY_CHECKERS,
            *MODULAR_IDENTITY_CHECKERS,
        ),
    )
