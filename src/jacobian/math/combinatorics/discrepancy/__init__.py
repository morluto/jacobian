"""Native exact discrepancy-theory operations and canonical values."""

from jacobian.math.combinatorics.discrepancy._models import (
    FiniteSetSystem,
    HardConstraintRoundingSource,
)
from jacobian.math.combinatorics.discrepancy.operations import (
    compute_discrepancy,
    compute_hard_constraint_rounding,
    compute_optimal_discrepancy,
)

__all__ = [
    "FiniteSetSystem",
    "HardConstraintRoundingSource",
    "compute_discrepancy",
    "compute_hard_constraint_rounding",
    "compute_optimal_discrepancy",
]
