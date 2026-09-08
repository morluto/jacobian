"""Execute admitted exact Levi-Civita covariant-derivative plans."""

from __future__ import annotations

import time

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.differential._recognition_process import (
    RationalFunctionRecognitionCandidate,
    gcd_recognition_values,
    recognize_canonical_rational_functions,
)
from jacobian.math.geometry.differential.metrics._dag import admit_recognition_work
from jacobian.math.geometry.differential.metrics._models import (
    RationalCoordinateMetric,
)
from jacobian.math.geometry.differential.rational_tensor.covariant_derivative._plan import (
    _covariant_reject,
    build_plan,
)
from jacobian.math.geometry.differential.rational_tensor.covariant_derivative._process import (
    evaluate_admitted_covariant_derivative,
)
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    canonical_locus_guards,
)


def covariant_derivative(
    metric: RationalCoordinateMetric,
    tensor: RationalCoordinateTensor,
) -> RationalCoordinateTensor:
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
    sources = gcd_recognition_values((*metric.tensor.components, *tensor.components))
    admit_recognition_work(
        sources,
        reject=_covariant_reject,
        label="covariant derivative",
    )
    candidates = tuple(
        RationalFunctionRecognitionCandidate(
            owner="tensor", component=index, value=source
        )
        for index, source in enumerate(sources)
    )
    if candidates:
        recognition = recognize_canonical_rational_functions(
            candidates, deadline=deadline
        )
        if recognition.non_coprime is not None:
            raise OperationDomainValidationError(
                location=("covariant_derivative",),
                code="differential_geometry.covariant_derivative.noncanonical_source",
                message=(
                    "metric and tensor components must be reduced canonical "
                    "rational functions"
                ),
            )
    plan = build_plan(metric, tensor)
    request_checkpoint("after covariant-derivative admission")
    axis = metric.tensor.coordinate_axis
    components, determinant_guards = evaluate_admitted_covariant_derivative(
        plan,
        axis,
        sources=(),
        deadline=deadline,
    )
    guards = canonical_locus_guards(
        metric.tensor.retained_nonzero_denominators,
        tensor.retained_nonzero_denominators,
        determinant_guards,
        component_denominators=tuple(value.denominator for value in components),
        variable_count=len(axis),
    )
    request_checkpoint("after covariant-derivative result construction")
    return RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=("COVARIANT", *tensor.variance),
        components=components,
        retained_nonzero_denominators=guards,
    )


__all__ = ["covariant_derivative"]
