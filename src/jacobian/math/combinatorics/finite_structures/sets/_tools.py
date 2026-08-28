"""Exact finite-integer-set operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.combinatorics.finite_structures.sets._set_cardinality import (
    SET_CARDINALITY_OPERATIONS,
)
from jacobian.math.combinatorics.finite_structures.sets._set_operations import (
    SET_OPERATION_OPERATIONS,
)
from jacobian.math.combinatorics.finite_structures.sets._set_predicates import (
    SET_PREDICATE_OPERATIONS,
)

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    *SET_OPERATION_OPERATIONS,
    *SET_PREDICATE_OPERATIONS,
    *SET_CARDINALITY_OPERATIONS,
)
