"""Exact combinatorics operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["combinatorics_operations"]


def combinatorics_operations() -> MathTools:
    from jacobian.domains.combinatorics.counting import COUNTING_OPERATIONS
    from jacobian.domains.combinatorics.difference_sets import DIFFERENCE_SET_OPERATIONS
    from jacobian.domains.combinatorics.partitions import PARTITION_OPERATIONS
    from jacobian.domains.combinatorics.recurrence import RECURRENCE_OPERATIONS

    return (
        *COUNTING_OPERATIONS,
        *PARTITION_OPERATIONS,
        *RECURRENCE_OPERATIONS,
        *DIFFERENCE_SET_OPERATIONS,
    )
