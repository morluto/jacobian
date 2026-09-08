"""Exact rational metric curvature operation declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.differential.metrics._models import (
    RationalMetricCurvatureProfile,
    RationalMetricCurvatureRequest,
)
from jacobian.math.geometry.differential.metrics.operations import curvature_profile


def _compute(request: RationalMetricCurvatureRequest) -> RationalMetricCurvatureProfile:
    return curvature_profile(request.metric)


_ONE = {"terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0, 0]}]}
_R_SQUARED = {"terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [2, 0]}]}
_ZERO: dict[str, list[object]] = {"terms": []}

TOOLS = (
    MathTool(
        operation_id="differential_geometry.rational_metric.curvature_profile.compute",
        title="Compute exact curvature of a rational coordinate metric",
        description=(
            "Compute the inverse metric, Levi-Civita connection, Riemann tensor "
            "R^l_kij, Ricci tensor and scalar curvature of a symmetric rational "
            "metric on 1..4 ordered coordinates. Convention: "
            "[nabla_i,nabla_j]v^l=R^l_kij v^k and Ric_kj=sum_i R^i_kij. "
            "Retain the source and complete nondegenerate chart locus, even "
            "when normalized curvature vanishes. Accept bounded exact "
            "polynomial DAGs; reject singular metrics and excessive symbolic "
            "growth. No positivity or global manifold assertion is made."
        ),
        request_type=RationalMetricCurvatureRequest,
        result_type=RationalMetricCurvatureProfile,
        run=_compute,
        tags=(
            "differential-geometry",
            "metric",
            "curvature",
            "riemann",
            "ricci",
            "rational",
        ),
        examples=(
            OperationExample(
                name="flat_polar_chart",
                description="The polar Euclidean metric has nonzero connection and zero curvature on r!=0.",
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
                    }
                },
            ),
        ),
    ),
)
