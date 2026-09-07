"""Publication of exact chordal sparse PSD decomposition."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.matrices.chordal_psd._models import (
    ChordalPSDDecomposition,
    ChordalPSDRequest,
)
from jacobian.math.matrices.chordal_psd.operations import decompose_chordal_psd


def _run(request: ChordalPSDRequest) -> ChordalPSDDecomposition:
    return decompose_chordal_psd(request.matrix, request.graph)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="matrix.chordal_psd.decompose",
        title="Decompose a chordal sparse PSD matrix into rational clique terms",
        description="Return rational PSD local matrices with original-axis inclusions whose embedded sum equals the supplied fully specified matrix. The graph must be chordal and cover every nonzero off-diagonal entry; extra zero-valued graph edges are allowed. Zero pivots emit no term, including an empty sum for the zero matrix. This constructs allocated clique summands, not a partial-matrix completion. Graph work, local output, and rational minor-height budgets bound execution.",
        request_type=ChordalPSDRequest,
        result_type=ChordalPSDDecomposition,
        run=_run,
        tags=(
            "matrix",
            "chordal",
            "positive semidefinite",
            "sparse",
            "clique",
            "decomposition",
            "LDL",
            "exact",
        ),
        examples=(
            OperationExample(
                name="path_psd_sum",
                description="Decompose the singular path matrix into two edge-supported PSD terms; the symmetric PSD matrix must have zeros outside its chordal graph.",
                input={
                    "matrix": {
                        "domain": "QQ",
                        "entries": [
                            [{"num": str(x), "den": "1"} for x in row]
                            for row in ((1, 1, 0), (1, 2, 1), (0, 1, 1))
                        ],
                    },
                    "graph": {"vertex_count": 3, "edges": [[0, 1], [1, 2]]},
                },
            ),
        ),
    ),
)
