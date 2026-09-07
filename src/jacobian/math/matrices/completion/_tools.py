"""Exact chordal PSD completion declaration."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.matrices.completion._models import (
    ChordalPSDCompletionRequest,
    ChordalPSDCompletionResult,
)
from jacobian.math.matrices.completion.operations import complete_chordal_psd


def _run(request: ChordalPSDCompletionRequest) -> ChordalPSDCompletionResult:
    return complete_chordal_psd(request.matrix)


TOOLS: MathTools = (
    MathTool(
        operation_id="matrix.chordal_psd_completion.compute",
        title="Complete a chordal partial rational matrix to PSD",
        description=(
            "Construct one exact rational positive-semidefinite completion of a "
            "partial symmetric matrix with chordal specified pattern and all "
            "diagonals. Missing entries are distinct from specified zeros. "
            "Singular separators are supported. Returns the completed QQ matrix "
            "and its partial source, or a specified non-PSD principal clique. "
            "Nonchordal patterns are unsupported, never a nonexistence verdict. "
            "Deterministic MCS/RREF construction; no optimization is promised. "
            "Admits recognition, clique solves, rational growth and dense output "
            "(at most 65,536 entries) under a shared 30-second safety deadline."
        ),
        request_type=ChordalPSDCompletionRequest,
        result_type=ChordalPSDCompletionResult,
        run=_run,
        tags=("matrix", "chordal", "positive-semidefinite", "completion", "exact"),
        examples=(
            OperationExample(
                name="missing_path_entry",
                description="Complete two correlated edges; the missing entry becomes 16/25.",
                input={
                    "matrix": {
                        "graph": {"vertex_count": 3, "edges": [[0, 1], [1, 2]]},
                        "specified_entries": [
                            {"row": 0, "column": 0, "value": {"num": "1", "den": "1"}},
                            {"row": 0, "column": 1, "value": {"num": "4", "den": "5"}},
                            {"row": 1, "column": 1, "value": {"num": "1", "den": "1"}},
                            {"row": 1, "column": 2, "value": {"num": "4", "den": "5"}},
                            {"row": 2, "column": 2, "value": {"num": "1", "den": "1"}},
                        ],
                    }
                },
            ),
        ),
    ),
)
