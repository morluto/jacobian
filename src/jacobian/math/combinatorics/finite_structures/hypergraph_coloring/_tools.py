"""Non-monochromatic colouring operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def hc_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: MathTools = (
    hc_operation(
        "hypergraph.nonmonochromatic_vertex_coloring.q_decide",
        "Decide hypergraph non-monochromatic q-colourability",
        (
            "Given a finite hypergraph H and a positive palette size q, decide "
            "whether H has a vertex q-colouring in which no hyperedge is "
            "monochromatic. Returns COLORABLE with one witness colouring, or "
            "NOT_COLORABLE."
        ),
        NonmonochromaticColoringRequest,
        NonmonochromaticColoringResult,
        compute_nonmonochromatic_coloring,
        "hypergraph",
        "coloring",
        "exact",
        examples=(
            example(
                "colorable_3edge",
                "A 3-edge hypergraph on {0,1,2} with q=2 is colourable.",
                {
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
