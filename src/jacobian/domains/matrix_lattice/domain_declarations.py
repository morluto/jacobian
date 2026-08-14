"""Exact matrix operation declarations."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.matrix_lattice.checkers import MATRIX_AUTHORIZED_CHECKERS
from jacobian.domains.matrix_lattice.hnf import HERMITE_NORMAL_FORM_OPERATION
from jacobian.domains.matrix_lattice.operation_declarations import MATRIX_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def matrix_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        (*MATRIX_OPERATIONS, HERMITE_NORMAL_FORM_OPERATION),
        OperationDiagnostic(
            code="INVALID_EXACT_MATRIX_REQUEST",
            stage="matrix_input_validation",
            message="Input does not satisfy the bounded exact matrix contract.",
            hint=(
                "Use a nonempty 1..32 by 1..32 matrix with canonical QQ or ZZ "
                "entries of at most 256 decimal digits."
            ),
        ),
    )


AUTHORIZED_CHECKERS = MATRIX_AUTHORIZED_CHECKERS
