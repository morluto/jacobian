"""Exact rational-polynomial operation declarations."""

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.polynomial.checkers import POLYNOMIAL_AUTHORIZED_CHECKERS
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
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def polynomial_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        (
            *POLYNOMIAL_INVARIANT_OPERATIONS,
            POLYNOMIAL_GROEBNER_OPERATION,
            GRADED_JACOBIAN_SYZYGY_OPERATION,
            JACOBIAN_SYZYGY_COEFFICIENT_LEDGER_OPERATION,
            *INTEGER_POLYNOMIAL_OPERATIONS,
            *RATIONAL_POLYNOMIAL_OPERATIONS,
        ),
        OperationDiagnostic(
            code="INVALID_POLYNOMIAL_REQUEST",
            stage="polynomial_input_validation",
            message="Input does not satisfy the bounded rational-polynomial contract.",
            hint="Use canonical sparse QQ polynomials and inspect the operation limits.",
        ),
    )


__all__ = ["polynomial_operations"]

AUTHORIZED_CHECKERS = POLYNOMIAL_AUTHORIZED_CHECKERS
