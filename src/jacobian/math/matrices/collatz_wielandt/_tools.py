"""Typed declarations for the Collatz-Wielandt quotient profile operation."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.matrices.collatz_wielandt._models import (
    CollatzWielandtRequest,
    CollatzWielandtResult,
)
from jacobian.math.matrices.collatz_wielandt.operations import (
    compute_collatz_wielandt_profile,
)


def _compute(request: CollatzWielandtRequest) -> CollatzWielandtResult:
    return compute_collatz_wielandt_profile(request.matrix, request.vector)


TOOLS: MathTools = (
    MathTool(
        operation_id="matrix.collatz_wielandt.quotient_profile.compute",
        title="Compute nonnegative Collatz-Wielandt upper quotient profiles",
        description=(
            "Given a nonnegative square matrix A and a strictly positive "
            "vector x, return the componentwise quotient profile (Ax)_i / x_i "
            "and its maximum, which is an exact upper bound on the spectral "
            "radius."
        ),
        request_type=CollatzWielandtRequest,
        result_type=CollatzWielandtResult,
        run=_compute,
        tags=("matrix", "collatz", "wielandt", "exact"),
        examples=(
            OperationExample(
                name="identity",
                description="Identity matrix with uniform vector.",
                input={
                    "matrix": [
                        [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                        [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                    ],
                    "vector": [
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
