"""Publication of exact one-parameter polynomial matrix inertia cells."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.matrices.inertia_cells._models import (
    InertiaCellsRequest,
    InertiaCellsResult,
)
from jacobian.math.matrices.inertia_cells.operations import compute_inertia_cells


def _run(request: InertiaCellsRequest) -> InertiaCellsResult:
    return compute_inertia_cells(request.matrix, request.interval)


TOOLS: MathTools = (
    MathTool(
        operation_id="matrix.polynomial.inertia_cells.compute",
        title="Compute exact inertia cells of a symmetric polynomial matrix",
        description="Partition a closed rational parameter interval into alternating exact point cells and open intervals on which a symmetric QQ[t] matrix has constant inertia. Retains every rank-drop specialization, including persistent nullspaces and tangencies. Algebraic boundaries retain canonical root identities and rational isolating intervals. Both endpoints are singleton cells; a singleton domain has one point cell. Connected support blocks are processed independently under characteristic-growth, root-isolation, sign-work and output bounds with a shared 60-second deadline.",
        request_type=InertiaCellsRequest,
        result_type=InertiaCellsResult,
        run=_run,
        tags=("matrix", "polynomial", "inertia", "parameter", "singular", "exact"),
        examples=(
            OperationExample(
                name="persistent_nullspace",
                description="Partition diag(t²-2,0) at both irrational singular parameters; the polynomial matrix must be symmetric and the rational interval closed.",
                input={
                    "matrix": {
                        "variables": ["t"],
                        "row_count": 2,
                        "column_count": 2,
                        "entries": [
                            [
                                {
                                    "variables": ["t"],
                                    "polynomial": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "1", "den": "1"},
                                                "exponents": [2],
                                            },
                                            {
                                                "coefficient": {
                                                    "num": "-2",
                                                    "den": "1",
                                                },
                                                "exponents": [0],
                                            },
                                        ]
                                    },
                                },
                                {"variables": ["t"], "polynomial": {"terms": []}},
                            ],
                            [
                                {"variables": ["t"], "polynomial": {"terms": []}},
                                {"variables": ["t"], "polynomial": {"terms": []}},
                            ],
                        ],
                    },
                    "interval": {
                        "lower": {"num": "-2", "den": "1"},
                        "upper": {"num": "2", "den": "1"},
                    },
                },
            ),
        ),
    ),
)
