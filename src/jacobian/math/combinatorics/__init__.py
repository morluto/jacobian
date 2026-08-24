"""Provider-independent exact combinatorics values and functions."""

from jacobian.math.combinatorics.exact_cover import (
    ExactCoverItemMultiplicity,
    ExactCoverRow,
    ExactCoverSearchStatus,
    GeneralizedExactCoverInstance,
    GeneralizedExactCoverRequest,
    GeneralizedExactCoverResult,
    find_generalized_exact_cover,
)
from jacobian.math.combinatorics.operations import (
    bell_number,
    bernoulli_number,
    catalan_number,
    derangement_number,
    double_factorial,
    fibonacci_number,
    integer_partitions,
    lucas_number,
    motzkin_number,
    partition_number,
    stirling_first,
    stirling_second,
)
from jacobian.math.combinatorics.recurrence_tables import (
    IndexedRecurrenceResidual,
    PolynomialCoefficientRecurrenceTableRequest,
    PolynomialCoefficientRecurrenceTableResult,
    recurrence_table_residuals,
)

__all__ = [
    "ExactCoverItemMultiplicity",
    "ExactCoverRow",
    "ExactCoverSearchStatus",
    "GeneralizedExactCoverInstance",
    "GeneralizedExactCoverRequest",
    "GeneralizedExactCoverResult",
    "IndexedRecurrenceResidual",
    "PolynomialCoefficientRecurrenceTableRequest",
    "PolynomialCoefficientRecurrenceTableResult",
    "bell_number",
    "bernoulli_number",
    "catalan_number",
    "derangement_number",
    "double_factorial",
    "fibonacci_number",
    "find_generalized_exact_cover",
    "integer_partitions",
    "lucas_number",
    "motzkin_number",
    "partition_number",
    "recurrence_table_residuals",
    "stirling_first",
    "stirling_second",
]
