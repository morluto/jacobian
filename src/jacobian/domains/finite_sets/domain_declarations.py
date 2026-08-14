"""Exact finite-integer-set operation declarations."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.finite_sets.set_cardinality import SET_CARDINALITY_OPERATIONS
from jacobian.domains.finite_sets.set_operations import SET_OPERATION_OPERATIONS
from jacobian.domains.finite_sets.set_predicates import SET_PREDICATE_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def finite_set_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        (
            *SET_OPERATION_OPERATIONS,
            *SET_PREDICATE_OPERATIONS,
            *SET_CARDINALITY_OPERATIONS,
        ),
        OperationDiagnostic(
            code="INVALID_FINITE_SET_REQUEST",
            stage="finite_set_input_validation",
            message="Input does not satisfy the finite-integer-set contract.",
            hint="Use canonical integer strings and inspect the operation's set schema.",
        ),
    )
