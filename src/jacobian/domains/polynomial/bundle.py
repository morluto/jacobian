"""Installation bundle for exact rational polynomial operations."""

from jacobian.domains.polynomial.checkers import POLYNOMIAL_EXACT_REPLAY_CHECKERS
from jacobian.domains.polynomial.elementary import (
    INTEGER_POLYNOMIAL_OPERATIONS,
    RATIONAL_POLYNOMIAL_OPERATIONS,
)
from jacobian.domains.polynomial.groebner import POLYNOMIAL_GROEBNER_OPERATION
from jacobian.domains.polynomial.invariants import (
    POLYNOMIAL_INVARIANT_OPERATIONS,
)
from jacobian.domains.polynomial.jacobian_syzygy import (
    GRADED_JACOBIAN_SYZYGY_OPERATION,
    JACOBIAN_SYZYGY_COEFFICIENT_LEDGER_OPERATION,
)
from jacobian.operation_declarations import OperationDeclarations


def build_polynomial_bundle() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return (
        *POLYNOMIAL_INVARIANT_OPERATIONS,
        POLYNOMIAL_GROEBNER_OPERATION,
        GRADED_JACOBIAN_SYZYGY_OPERATION,
        JACOBIAN_SYZYGY_COEFFICIENT_LEDGER_OPERATION,
        *INTEGER_POLYNOMIAL_OPERATIONS,
        *RATIONAL_POLYNOMIAL_OPERATIONS,
    )


__all__ = ["build_polynomial_bundle"]

CHECKER_DECLARATIONS = POLYNOMIAL_EXACT_REPLAY_CHECKERS
