"""Homogeneous progression set system operation declarations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.combinatorics.discrepancy._models import FiniteSetSystem
from jacobian.math.combinatorics.discrepancy.homogeneous_progression._models import (
    HomogeneousProgressionRequest,
)
from jacobian.math.combinatorics.discrepancy.homogeneous_progression.operations import (
    construct_homogeneous_progression_set_system,
)


def compute_homogeneous_progression(
    request: HomogeneousProgressionRequest,
) -> FiniteSetSystem:
    return construct_homogeneous_progression_set_system(request.n)


TOOLS: MathTools = (
    MathTool(
        "discrepancy.homogeneous_progression_set_system.construct",
        "Construct the homogeneous progression set system on [n]",
        (
            "Construct the canonical finite set system whose ground set is [n] "
            "and whose sets are the homogeneous arithmetic progressions "
            "{d, 2d, ..., kd} for every d, k >= 1 with dk <= n, with zero-based "
            "indexing. This is the standard carrier for Erdős discrepancy "
            "experiments."
        ),
        HomogeneousProgressionRequest,
        FiniteSetSystem,
        compute_homogeneous_progression,
        tags=("combinatorics", "discrepancy", "exact"),
        examples=(
            example(
                "n4",
                "The homogeneous progression set system on [4].",
                {"n": 4},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
