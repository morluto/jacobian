"""Non-monochromatic colouring operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.finite_structures.hypergraph_coloring._models import (
    NonmonochromaticColoringRequest,
    NonmonochromaticColoringResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraph_coloring.operations import (
    decide_nonmonochromatic_coloring,
)


def compute_nonmonochromatic_coloring(
    request: NonmonochromaticColoringRequest,
) -> NonmonochromaticColoringResult:
    return decide_nonmonochromatic_coloring(request.hypergraph, request.palette_size)


TOOLS: MathTools = (
    MathTool(
        operation_id="hypergraph.nonmonochromatic_vertex_coloring.q_decide",
        title="Decide hypergraph non-monochromatic q-colourability",
        description=(
            "Given a finite hypergraph H and a positive palette size q, decide "
            "whether H has a vertex q-colouring in which no hyperedge is "
            "monochromatic. Returns COLORABLE with one witness colouring, or "
            "NOT_COLORABLE."
        ),
        request_type=NonmonochromaticColoringRequest,
        result_type=NonmonochromaticColoringResult,
        run=compute_nonmonochromatic_coloring,
        tags=("hypergraph", "coloring", "exact"),
        examples=(
            OperationExample(
                name="colorable_3edge",
                description="A 3-edge hypergraph on {0,1,2} with q=2 is colourable.",
                input={
                    "hypergraph": {
                        "vertices": ["0", "1", "2"],
                        "edges": [
                            ["e0", ["0", "1", "2"]],
                        ],
                    },
                    "palette_size": 2,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
