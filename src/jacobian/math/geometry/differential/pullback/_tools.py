"""Operation declaration for exact rational metric pullbacks."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.differential.pullback._models import (
    RationalMetricPullbackProfile,
    RationalMetricPullbackRequest,
)
from jacobian.math.geometry.differential.pullback.operations import pullback_metric


def _compute(request: RationalMetricPullbackRequest) -> RationalMetricPullbackProfile:
    return pullback_metric(request.metric, request.map)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="differential_geometry.rational_metric.pullback.compute",
        title="Pull back a rational coordinate metric",
        description="Compute the exact covariant pullback tensor and retain its source-bound construction locus.",
        request_type=RationalMetricPullbackRequest,
        result_type=RationalMetricPullbackProfile,
        run=_compute,
        tags=("differential-geometry", "metric", "pullback", "rational"),
        examples=(
            OperationExample(
                name="square_map_on_a_line",
                description="Pull back du² along u=x² to obtain 4x² dx², retaining the valid degenerate point x=0.",
                input={
                    "metric": {
                        "tensor": {
                            "coordinate_axis": ["u"],
                            "variance": ["COVARIANT", "COVARIANT"],
                            "components": [
                                {
                                    "variables": ["u"],
                                    "numerator": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "1", "den": "1"},
                                                "exponents": [0],
                                            }
                                        ]
                                    },
                                    "denominator": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "1", "den": "1"},
                                                "exponents": [0],
                                            }
                                        ]
                                    },
                                }
                            ],
                        }
                    },
                    "map": {
                        "source_variables": ["x"],
                        "target_coordinates": ["u"],
                        "components": [
                            {
                                "variables": ["x"],
                                "numerator": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [2],
                                        }
                                    ]
                                },
                                "denominator": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [0],
                                        }
                                    ]
                                },
                            }
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
