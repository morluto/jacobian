"""Homogeneous progression set system operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
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
        operation_id="discrepancy.homogeneous_progression_set_system.construct",
        title="Construct the homogeneous progression set system on [n]",
        description=(
            "Construct the canonical finite set system whose ground set is [n] "
            "and whose sets are the homogeneous arithmetic progressions "
            "{d, 2d, ..., kd} for every d, k >= 1 with dk <= n, with zero-based "
            "indexing. This is the standard carrier for Erdős discrepancy "
            "experiments."
        ),
        request_type=HomogeneousProgressionRequest,
        result_type=FiniteSetSystem,
        run=compute_homogeneous_progression,
        tags=("combinatorics", "discrepancy", "exact"),
        examples=(
            OperationExample(
                name="n4",
                description="The homogeneous progression set system on [4].",
                input={"n": 4},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
