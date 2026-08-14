"""Bounded exact finite-poset operation declarations."""

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.posets.checkers import FINITE_POSET_AUTHORIZED_CHECKERS
from jacobian.domains.posets.operations import FINITE_POSET_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def finite_poset_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        FINITE_POSET_OPERATIONS,
        OperationDiagnostic(
            code="INVALID_FINITE_POSET_REQUEST",
            stage="finite_poset_input_validation",
            message="Input does not satisfy the bounded finite-poset contract.",
            hint=(
                "Declare unique labels and either exact cover edges or the complete "
                "comparable-pair relation under an explicit reflexive policy."
            ),
        ),
    )


__all__ = ["finite_poset_operations"]

AUTHORIZED_CHECKERS = FINITE_POSET_AUTHORIZED_CHECKERS
