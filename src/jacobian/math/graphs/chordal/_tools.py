"""Chordal recognition operation declarations."""

from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationExample,
)
from jacobian.math.graphs.chordal._models import (
    ChordalRecognitionRequest,
    ChordalRecognitionResult,
)
from jacobian.math.graphs.chordal.operations import recognize_chordal


def _compute_chordal_recognition(
    request: ChordalRecognitionRequest,
) -> ChordalRecognitionResult:
    return recognize_chordal(request.graph)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.chordal.recognition.compute",
        title="Recognize chordal graphs with structural certificates",
        description=(
            "Decide chordality of a finite simple graph. CHORDAL returns a "
            "perfect elimination ordering (every vertex's later neighbors "
            "form a clique); NONCHORDAL returns an ordered induced cycle of "
            "length at least four. A failed candidate ordering alone never "
            "certifies non-chordality."
        ),
        request_type=ChordalRecognitionRequest,
        result_type=ChordalRecognitionResult,
        run=_compute_chordal_recognition,
        tags=("graph", "chordal", "exact"),
        examples=(
            OperationExample(
                name="path_is_chordal",
                description="Recognize the 3-vertex path as chordal.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"]],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
