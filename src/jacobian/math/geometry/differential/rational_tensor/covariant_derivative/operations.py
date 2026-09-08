"""Execute admitted exact Levi-Civita covariant-derivative plans."""

from __future__ import annotations

import time

from pydantic_core import PydanticCustomError

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.differential.metrics._models import (
    RationalCoordinateMetric,
)
from jacobian.math.geometry.differential.rational_tensor.covariant_derivative._models import (
    RationalCovariantDerivativeProfile,
)
from jacobian.math.geometry.differential.rational_tensor.covariant_derivative._plan import (
    build_plan,
)
from jacobian.math.geometry.differential.rational_tensor.covariant_derivative._process import (
    evaluate_admitted_covariant_derivative,
)
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    canonical_locus_guards,
)
from jacobian.math.polynomials.values import (
    require_canonical_rational_function,
)


def _recognize_sources(
    metric: RationalCoordinateMetric, tensor: RationalCoordinateTensor
) -> None:
    for source in (*metric.tensor.components, *tensor.components):
        request_checkpoint("before covariant-derivative source recognition")
        try:
            require_canonical_rational_function(source)
        except PydanticCustomError as exc:
            raise OperationDomainValidationError(
                location=("covariant_derivative",),
                code="differential_geometry.covariant_derivative.noncanonical_source",
                message="metric and tensor components must be reduced canonical rational functions",
            ) from exc


def covariant_derivative(
    metric: RationalCoordinateMetric,
    tensor: RationalCoordinateTensor,
) -> RationalCovariantDerivativeProfile:
    """Return the exact covariant derivative with one leading covariant axis."""
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return covariant_derivative(metric, tensor)
    deadline = execution.started_at + 120.0
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("before covariant-derivative admission")
    if tensor.coordinate_axis != metric.tensor.coordinate_axis:
        raise OperationDomainValidationError(
            location=("covariant_derivative",),
            code="differential_geometry.covariant_derivative.axis_mismatch",
            message="metric and tensor must use the same coordinate axis",
        )
    plan = build_plan(metric, tensor)
    request_checkpoint("after covariant-derivative admission")
    _recognize_sources(metric, tensor)
    axis = metric.tensor.coordinate_axis
    components, determinant_guards = evaluate_admitted_covariant_derivative(
        plan, axis, deadline=deadline
    )
    guards = canonical_locus_guards(
        metric.tensor.retained_nonzero_denominators,
        tensor.retained_nonzero_denominators,
        determinant_guards,
        component_denominators=tuple(value.denominator for value in components),
        variable_count=len(axis),
    )
    result = RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=("COVARIANT", *tensor.variance),
        components=components,
        retained_nonzero_denominators=guards,
    )
    request_checkpoint("after covariant-derivative result construction")
    return RationalCovariantDerivativeProfile._from_kernel(
        metric=metric, source=tensor, covariant_derivative=result
    )


__all__ = ["covariant_derivative"]
