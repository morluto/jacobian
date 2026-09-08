"""Operation declaration for exact Laplace--Beltrami values."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.differential.laplace_beltrami._models import (
    RationalLaplaceBeltramiRequest,
    RationalLaplaceBeltramiResult,
)
from jacobian.math.geometry.differential.laplace_beltrami.operations import (
    laplace_beltrami,
)


def _compute(request: RationalLaplaceBeltramiRequest) -> RationalLaplaceBeltramiResult:
    return laplace_beltrami(request.metric, request.scalar)


_ONE = {"terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}]}
_X_SQUARED = {"terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [2]}]}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="differential_geometry.rational_metric.laplace_beltrami.compute",
        title="Compute an exact rational Laplace--Beltrami value",
        description=(
            "Contract the rational scalar Hessian with the inverse metric using "
            "the metric's Levi-Civita connection and preserve exact chart semantics."
        ),
        request_type=RationalLaplaceBeltramiRequest,
        result_type=RationalLaplaceBeltramiResult,
        run=_compute,
        tags=("differential-geometry", "metric", "laplace-beltrami", "rational"),
        examples=(
            OperationExample(
                name="one_dimensional_square",
                description="The one-dimensional unit metric sends x^2 to 2.",
                input={
                    "metric": {
                        "tensor": {
                            "coordinate_axis": ["x"],
                            "variance": ["COVARIANT", "COVARIANT"],
                            "components": [
                                {
                                    "variables": ["x"],
                                    "numerator": _ONE,
                                    "denominator": _ONE,
                                }
                            ],
                        }
                    },
                    "scalar": {
                        "variables": ["x"],
                        "numerator": _X_SQUARED,
                        "denominator": _ONE,
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
