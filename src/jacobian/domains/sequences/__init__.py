"""Exact finite integer-sequence operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["sequence_operations"]


def sequence_operations() -> MathTools:
    from jacobian.domains.sequences.aggregates import SEQUENCE_AGGREGATE_OPERATIONS
    from jacobian.domains.sequences.predicates import SEQUENCE_PREDICATE_OPERATIONS
    from jacobian.domains.sequences.search import SEQUENCE_SEARCH_OPERATIONS
    from jacobian.domains.sequences.statistics import SEQUENCE_STATISTIC_OPERATIONS
    from jacobian.domains.sequences.transforms import SEQUENCE_TRANSFORM_OPERATIONS

    return (
        *SEQUENCE_AGGREGATE_OPERATIONS,
        *SEQUENCE_STATISTIC_OPERATIONS,
        *SEQUENCE_TRANSFORM_OPERATIONS,
        *SEQUENCE_PREDICATE_OPERATIONS,
        *SEQUENCE_SEARCH_OPERATIONS,
    )
