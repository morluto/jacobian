"""Exact finite integer-sequence operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.number_theory.sequences.core._aggregates import (
    SEQUENCE_AGGREGATE_OPERATIONS,
)
from jacobian.math.number_theory.sequences.core._predicates import (
    SEQUENCE_PREDICATE_OPERATIONS,
)
from jacobian.math.number_theory.sequences.core._search import (
    SEQUENCE_SEARCH_OPERATIONS,
)
from jacobian.math.number_theory.sequences.core._statistics import (
    SEQUENCE_STATISTIC_OPERATIONS,
)
from jacobian.math.number_theory.sequences.core._transforms import (
    SEQUENCE_TRANSFORM_OPERATIONS,
)

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    *SEQUENCE_AGGREGATE_OPERATIONS,
    *SEQUENCE_STATISTIC_OPERATIONS,
    *SEQUENCE_TRANSFORM_OPERATIONS,
    *SEQUENCE_PREDICATE_OPERATIONS,
    *SEQUENCE_SEARCH_OPERATIONS,
)
