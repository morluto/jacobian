"""Operation declaration for exact rational covariant derivatives."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.differential.rational_tensor.covariant_derivative._models import (
    RationalCovariantDerivativeRequest,
)
from jacobian.math.geometry.differential.rational_tensor.covariant_derivative.operations import (
    covariant_derivative,
)
from jacobian.math.geometry.differential.values import RationalCoordinateTensor


def _compute(
    request: RationalCovariantDerivativeRequest,
) -> RationalCoordinateTensor:
    return covariant_derivative(request.metric, request.tensor)


_ONE = {"terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0, 0]}]}
_ZERO: dict[str, list[object]] = {"terms": []}
_R_SQUARED = {"terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [2, 0]}]}

TOOLS = (
    MathTool(
        operation_id="differential_geometry.rational_tensor.covariant_derivative.compute",
        title="Compute an exact rational covariant derivative",
        description=(
            "Apply the Levi-Civita connection of a rational coordinate metric "
            "to a rational mixed tensor. Return the derivative tensor with one "
            "leading covariant index and the complete retained locus. "
            "The metric and tensor share the same ordered coordinate axis. "
            "For T^(i...)_(j...), use +Gamma on contravariant and -Gamma on "
            "covariant source indices."
        ),
        request_type=RationalCovariantDerivativeRequest,
        result_type=RationalCoordinateTensor,
        run=_compute,
        tags=("differential-geometry", "tensor", "covariant-derivative", "rational"),
        examples=(
            OperationExample(
                name="polar_metric_scalar",
                description=(
                    "Differentiate r^2 covariantly in the polar Euclidean chart; "
                    "the metric and scalar tensor must use the same ordered "
                    "(r, theta) coordinate axis."
                ),
                input={
                    "metric": {
                        "tensor": {
                            "coordinate_axis": ["r", "theta"],
                            "variance": ["COVARIANT", "COVARIANT"],
                            "components": [
                                {
                                    "variables": ["r", "theta"],
                                    "numerator": numerator,
                                    "denominator": _ONE,
                                }
                                for numerator in (_ONE, _ZERO, _ZERO, _R_SQUARED)
                            ],
                        }
                    },
                    "tensor": {
                        "coordinate_axis": ["r", "theta"],
                        "variance": [],
                        "components": [
                            {
                                "variables": ["r", "theta"],
                                "numerator": _R_SQUARED,
                                "denominator": _ONE,
                            }
                        ],
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
