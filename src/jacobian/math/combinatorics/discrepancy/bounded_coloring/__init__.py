"""Per-set bounded discrepancy decisions on canonical finite set systems."""

from jacobian.math.combinatorics.discrepancy.bounded_coloring._models import (
    BoundedColoringBudget,
    BoundedColoringResult,
)
from jacobian.math.combinatorics.discrepancy.bounded_coloring.operations import decide

__all__ = ["BoundedColoringBudget", "BoundedColoringResult", "decide"]
