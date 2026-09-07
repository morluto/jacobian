"""One supplied-relation semidefinite reduction."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.matrices.semidefinite._models import SemidefiniteFaceReductionRequest
from jacobian.math.matrices.semidefinite.operations import reduce_exposed_face
from jacobian.math.matrices.semidefinite.values import SemidefiniteFaceReduction


def compute_face_reduction(
    request: SemidefiniteFaceReductionRequest,
) -> SemidefiniteFaceReduction:
    return reduce_exposed_face(request.system, request.multipliers)


TOOLS: MathTools = (
    MathTool(
        operation_id="matrix.semidefinite.exposed_face.reduce",
        title="Reduce a rational semidefinite system to one exposed face",
        description=(
            "Given tr(A_i X)=b_i, X PSD and supplied rational y, recognize that "
            "W=sum y_i A_i is nonzero PSD and y^T b=0. Return equivalent equalities "
            "on ker(W) with the rational embedding X=VYV^T, retaining all equalities "
            "including contradictions and zero-dimensional faces. Bounded by "
            "dense cells, rational growth, output digits and elimination bit-work."
        ),
        request_type=SemidefiniteFaceReductionRequest,
        result_type=SemidefiniteFaceReduction,
        run=compute_face_reduction,
        tags=("matrix", "semidefinite", "facial-reduction", "exact"),
        discovery_terms=(
            "exposed face",
            "SDP reduction",
            "exposing relation",
            "Gram matrix face",
        ),
        examples=(
            OperationExample(
                name="one_dimensional_face",
                description="The relation X_00=0 forces the first PSD row and column to vanish.",
                input={
                    "system": {
                        "order": 2,
                        "matrices": [
                            {
                                "entries": [
                                    [
                                        {"num": "1", "den": "1"},
                                        {"num": "0", "den": "1"},
                                    ],
                                    [
                                        {"num": "0", "den": "1"},
                                        {"num": "0", "den": "1"},
                                    ],
                                ]
                            },
                        ],
                        "rhs": [{"num": "0", "den": "1"}],
                    },
                    "multipliers": [{"num": "1", "den": "1"}],
                },
            ),
        ),
    ),
)
