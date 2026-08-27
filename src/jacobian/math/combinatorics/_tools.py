"""Exact combinatorics operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.combinatorics._counting import COUNTING_OPERATIONS
from jacobian.math.combinatorics._difference_sets import DIFFERENCE_SET_OPERATIONS
from jacobian.math.combinatorics._exact_cover import GENERALIZED_EXACT_COVER_OPERATION
from jacobian.math.combinatorics._partitions import PARTITION_OPERATIONS
from jacobian.math.combinatorics._progression_hypergraph import (
    PROGRESSION_HYPERGRAPH_OPERATION,
)
from jacobian.math.combinatorics._recurrence import RECURRENCE_OPERATIONS
from jacobian.math.combinatorics._sidon_extension import SIDON_EXTENSION_OPERATION

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    *COUNTING_OPERATIONS,
    *PARTITION_OPERATIONS,
    *RECURRENCE_OPERATIONS,
    *DIFFERENCE_SET_OPERATIONS,
    GENERALIZED_EXACT_COVER_OPERATION,
    SIDON_EXTENSION_OPERATION[0],
    PROGRESSION_HYPERGRAPH_OPERATION,
)
