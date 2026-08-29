"""Typed declarations for the homogeneous progression set system constructor."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.combinatorics.discrepancy.homogeneous_progression._models import (
    HomogeneousProgressionRequest,
    HomogeneousProgressionResult,
)
from jacobian.math.combinatorics.discrepancy.homogeneous_progression.operations import (
    construct_homogeneous_progression_set_system,
)


def _construct(request: HomogeneousProgressionRequest) -> HomogeneousProgressionResult:
    return construct_homogeneous_progression_set_system(request.n)


TOOLS: MathTools = (
    MathTool(
        operation_id="discrepancy.homogeneous_progression_set_system.construct",
        title="Construct the finite homogeneous-progression set system",
        description=(
            "For a nonnegative integer n, return the canonical FiniteSetSystem "
            "whose ground set is 0..n-1 (representing 1..n) and whose sets are "
            "exactly the homogeneous arithmetic progressions (d, 2d, ..., kd) "
            "with dk <= n."
        ),
        request_type=HomogeneousProgressionRequest,
        result_type=HomogeneousProgressionResult,
        run=_construct,
        tags=("discrepancy", "combinatorics", "exact"),
        examples=(
            example(
                "n_6",
                "Homogeneous progression set system for n=6.",
                {"n": 6},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
