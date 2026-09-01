"""Cycle-length profile operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.cycle_length_profile._models import (
    CycleLengthProfileRequest,
    CycleLengthProfileResult,
)
from jacobian.math.graphs.cycle_length_profile.operations import (
    compute_cycle_length_profile,
)


def compute_cycle_length_profile_op(
    request: CycleLengthProfileRequest,
) -> CycleLengthProfileResult:
    return compute_cycle_length_profile(request.graph)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.invariant.cycle_length_profile.compute",
        title="Compute the complete cycle-length profile of a graph",
        description=(
            "Given a bounded finite simple graph, return the complete set of "
            "simple-cycle lengths together with one canonical witness cycle "
            "for each occurring length. Admission requires the first-witness "
            "search to fit 10,000,000 work units and bounds retained label "
            "characters independently of transport encoding."
        ),
        request_type=CycleLengthProfileRequest,
        result_type=CycleLengthProfileResult,
        run=compute_cycle_length_profile_op,
        tags=("graph", "invariant", "exact"),
        examples=(
            OperationExample(
                name="c4",
                description="C4 has cycle-length spectrum {4}.",
                input={
                    "graph": {
                        "vertices": ["0", "1", "2", "3"],
                        "edges": [["0", "1"], ["1", "2"], ["2", "3"], ["0", "3"]],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
