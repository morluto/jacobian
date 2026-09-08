"""Operation declaration for exact rational metric pullbacks."""

from typing import Any

from jacobian.catalog.models import MathTool
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
    ),
)

__all__ = ["TOOLS"]
