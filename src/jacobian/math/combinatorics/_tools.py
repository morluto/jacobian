"""Exact combinatorics operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.combinatorics._counting import COUNTING_OPERATIONS
from jacobian.math.combinatorics._difference_sets import DIFFERENCE_SET_OPERATIONS
from jacobian.math.combinatorics._partitions import PARTITION_OPERATIONS
from jacobian.math.combinatorics._recurrence import RECURRENCE_OPERATIONS

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    *COUNTING_OPERATIONS,
    *PARTITION_OPERATIONS,
    *RECURRENCE_OPERATIONS,
    *DIFFERENCE_SET_OPERATIONS,
)
