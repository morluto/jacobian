"""Bounded lattice-reduction operation declarations."""

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.matrix_lattice.lattice import (
    LATTICE_OPERATIONS,
)
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def lattice_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        LATTICE_OPERATIONS,
        OperationDiagnostic(
            code="INVALID_LATTICE_REDUCTION_REQUEST",
            stage="lattice_input_validation",
            message="Input does not satisfy the bounded exact lattice contract.",
            hint=(
                "Use a 1..32 by 1..32 canonical integer row basis with entries of "
                "at most 256 decimal digits."
            ),
        ),
    )
