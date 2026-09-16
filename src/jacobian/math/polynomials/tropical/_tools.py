"""Public declarations for exact tropical scalar arithmetic."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.polynomials.tropical._models import (
    ScalarAddRequest,
    ScalarAddResult,
)
from jacobian.math.polynomials.tropical.operations import tropical_scalar_add


def compute_scalar_add(request: ScalarAddRequest) -> ScalarAddResult:
    result, branch, infinity_case = tropical_scalar_add(
        request.semiring, request.left, request.right
    )
    return ScalarAddResult._from_kernel(
        request, result=result, branch=branch, infinity_case=infinity_case
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="tropical.scalar.add.compute",
        title="Add two scalars in an explicit tropical semiring",
        description=(
            "Return left tropical-plus right under one explicit MIN_PLUS "
            "(min) or MAX_PLUS (max) semiring over ZZ or QQ, with the winning "
            "branch and the finite/infinity case recorded. Only the infinity "
            "licensed by the chosen convention is an element; both operands "
            "must carry the request semiring."
        ),
        request_type=ScalarAddRequest,
        result_type=ScalarAddResult,
        run=compute_scalar_add,
        tags=("tropical", "semiring", "idempotent", "exact"),
        discovery_terms=(
            "min-plus addition",
            "max-plus addition",
            "tropical sum of scalars",
        ),
        examples=(
            OperationExample(
                name="min_plus_finite_sum",
                description=(
                    "Compute min(3, 5) = 3 in MIN_PLUS over ZZ; both operands "
                    "must carry the request semiring."
                ),
                input={
                    "semiring": {"convention": "MIN_PLUS", "base": "ZZ"},
                    "left": {
                        "semiring": {"convention": "MIN_PLUS", "base": "ZZ"},
                        "kind": "FINITE",
                        "value": {"num": "3", "den": "1"},
                    },
                    "right": {
                        "semiring": {"convention": "MIN_PLUS", "base": "ZZ"},
                        "kind": "FINITE",
                        "value": {"num": "5", "den": "1"},
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS", "compute_scalar_add"]
