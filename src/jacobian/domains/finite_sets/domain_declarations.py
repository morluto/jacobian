"""Exact finite-integer-set operation declarations."""

from __future__ import annotations

from jacobian.domains.finite_sets.set_cardinality import SET_CARDINALITY_OPERATIONS
from jacobian.domains.finite_sets.set_operations import SET_OPERATION_OPERATIONS
from jacobian.domains.finite_sets.set_predicates import SET_PREDICATE_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def finite_set_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return (
        *SET_OPERATION_OPERATIONS,
        *SET_PREDICATE_OPERATIONS,
        *SET_CARDINALITY_OPERATIONS,
    )


CHECKER_DECLARATIONS = ()
