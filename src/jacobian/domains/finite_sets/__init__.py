"""Exact finite-integer-set operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["finite_set_operations"]


def finite_set_operations() -> MathTools:
    from jacobian.domains.finite_sets.set_cardinality import SET_CARDINALITY_OPERATIONS
    from jacobian.domains.finite_sets.set_operations import SET_OPERATION_OPERATIONS
    from jacobian.domains.finite_sets.set_predicates import SET_PREDICATE_OPERATIONS

    return (
        *SET_OPERATION_OPERATIONS,
        *SET_PREDICATE_OPERATIONS,
        *SET_CARDINALITY_OPERATIONS,
    )
