"""Typed declarations for the cycle-length profile operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.cycle_length_profile._models import (
    CycleLengthProfileRequest,
    CycleLengthProfileResult,
)
from jacobian.math.graphs.cycle_length_profile.operations import (
    compute_cycle_length_profile,
)


def _compute(request: CycleLengthProfileRequest) -> CycleLengthProfileResult:
    return compute_cycle_length_profile(request.graph)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.invariant.cycle_length_profile.compute",
        title="Compute the complete simple-cycle length profile of a graph",
        description=(
            "For one bounded finite simple graph, return its complete set of "
            "simple-cycle lengths from 3 through |V|, with one canonical witness "
            "cycle for each occurring length."
        ),
        request_type=CycleLengthProfileRequest,
        result_type=CycleLengthProfileResult,
        run=_compute,
        tags=("graph", "cycle", "invariant", "exact"),
        examples=(
            example(
                "triangle",
                "The cycle length profile of a triangle.",
                {
 "graph": {"vertices": ["a", "b", "c"], "edges": [["a", "b"], ["b", "c"], ["a", "c"]]},
            },
        ),
    ),
)

__all__ = ["TOOLS"]
