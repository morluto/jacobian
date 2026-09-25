"""Public declaration of exact free-algebra homomorphism application."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.free_algebras.homomorphism._models import (
    FreeAlgebraHomomorphismApplyRequest,
    FreeAlgebraHomomorphismApplyResult,
)
from jacobian.math.free_algebras.homomorphism.operations import apply


def _run_apply(
    request: FreeAlgebraHomomorphismApplyRequest,
) -> FreeAlgebraHomomorphismApplyResult:
    return apply(request.homomorphism, request.polynomial)


TOOLS = (
    MathTool(
        operation_id="free_algebra.homomorphism.apply.compute",
        title="Apply a free associative algebra homomorphism",
        description=(
            "Substitute one exact target polynomial for each source generator in word order, "
            "extend multiplicatively and linearly over QQ, and collect a canonical target polynomial."
        ),
        request_type=FreeAlgebraHomomorphismApplyRequest,
        result_type=FreeAlgebraHomomorphismApplyResult,
        run=_run_apply,
        tags=("algebra", "free-algebra", "homomorphism", "noncommutative", "exact"),
        discovery_terms=(
            "free algebra homomorphism",
            "noncommutative substitution",
            "generator images",
        ),
        examples=(
            OperationExample(
                name="ordered_substitution",
                description="Map x to u+v and y to uv, then apply to xy-yx.",
                input={
                    "homomorphism": {
                        "source_alphabet": ["x", "y"],
                        "target_alphabet": ["u", "v"],
                        "generator_images": [
                            {
                                "alphabet": ["u", "v"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["v"],
                                    },
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["u"],
                                    },
                                ],
                            },
                            {
                                "alphabet": ["u", "v"],
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "word": ["u", "v"],
                                    }
                                ],
                            },
                        ],
                    },
                    "polynomial": {
                        "alphabet": ["x", "y"],
                        "terms": [
                            {
                                "coefficient": {"num": "-1", "den": "1"},
                                "word": ["y", "x"],
                            },
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "word": ["x", "y"],
                            },
                        ],
                    },
                },
            ),
        ),
    ),
)
