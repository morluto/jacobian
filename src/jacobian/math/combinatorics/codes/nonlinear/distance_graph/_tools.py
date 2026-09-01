"""Binary code distance graph operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.codes.nonlinear.distance_graph._models import (
    BinaryCodeDistanceGraphRequest,
    BinaryCodeDistanceGraphResult,
)
from jacobian.math.combinatorics.codes.nonlinear.distance_graph.operations import (
    compute_distance_graph,
)


def compute_distance_graph_op(
    request: BinaryCodeDistanceGraphRequest,
) -> BinaryCodeDistanceGraphResult:
    return compute_distance_graph(request.source, request.target_distance)


TOOLS: MathTools = (
    MathTool(
        operation_id="code.binary.explicit.distance_graph.compute",
        title="Construct the Hamming distance graph of a binary code",
        description=(
            "For a bounded explicit binary code and selected Hamming distance, "
            "return the complete indexed simple graph whose vertices are the "
            "canonical codeword indices and whose edges join exactly the "
            "distinct codeword pairs at that Hamming distance."
        ),
        request_type=BinaryCodeDistanceGraphRequest,
        result_type=BinaryCodeDistanceGraphResult,
        run=compute_distance_graph_op,
        tags=("coding-theory", "exact"),
        examples=(
            OperationExample(
                name="length3_dist1",
                description="Code {000, 011, 110} at distance 1.",
                input={
                    "source": {
                        "length": 3,
                        "codewords": [[0, 0, 0], [0, 1, 1], [1, 1, 0]],
                    },
                    "target_distance": 1,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
