"""Execute admitted exact Laplace--Beltrami plans."""

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
from jacobian.math.geometry.differential.laplace_beltrami._models import (
    RationalLaplaceBeltramiResult,
)
from jacobian.math.geometry.differential.laplace_beltrami._plan import (
    _laplace_metric_reject,
    _laplace_scalar_reject,
    build_plan,
)
from jacobian.math.geometry.differential.metrics._dag import admit_recognition_work
from jacobian.math.geometry.differential.metrics._dag_evaluate_process import (
    evaluate_admitted_dag,
)
from jacobian.math.geometry.differential.metrics._models import RationalCoordinateMetric
from jacobian.math.geometry.differential.values import canonical_locus_guards
from jacobian.math.polynomials.values import RationalFunction


def _singular_metric() -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("metric",),
        code="differential_geometry.laplace_beltrami.singular_metric",
        message="metric determinant is identically zero",
    )


def _recognition_location(
    failure: RationalFunctionRecognitionCandidate,
) -> tuple[str | int, ...]:
    if failure.owner == "scalar":
        return ("scalar",)
    return ("metric", "tensor", "components", failure.component)


def _recognize_source(
    metric: RationalCoordinateMetric, scalar: RationalFunction, deadline: float
) -> None:
    candidates: list[RationalFunctionRecognitionCandidate] = []
    seen: set[RationalFunction] = set()
    for index, component in enumerate(metric.tensor.components):
        if component in seen or not gcd_recognition_values((component,)):
            continue
        seen.add(component)
        candidates.append(
            RationalFunctionRecognitionCandidate(
                owner="metric", component=index, value=component
            )
        )
    if scalar not in seen and gcd_recognition_values((scalar,)):
        candidates.append(
            RationalFunctionRecognitionCandidate(
                owner="scalar", component=0, value=scalar
            )
        )
    owned = tuple(candidates)
    if not owned:
        return
    admit_recognition_work(
        tuple(candidate.value for candidate in owned),
        label="Laplace--Beltrami",
        reject_for=lambda value: (
            _laplace_scalar_reject if value == scalar else _laplace_metric_reject
        ),
    )
    recognition = recognize_canonical_rational_functions(owned, deadline=deadline)
    if recognition.non_coprime is not None:
        raise OperationDomainValidationError(
            location=_recognition_location(recognition.non_coprime),
            code="differential_geometry.laplace_beltrami.noncanonical_source",
            message="metric and scalar must be reduced canonical rational functions",
        )


def laplace_beltrami(
    metric: RationalCoordinateMetric, scalar: RationalFunction
) -> RationalLaplaceBeltramiResult:
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return laplace_beltrami(metric, scalar)
    deadline = execution.started_at + 120.0
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("before Laplace--Beltrami admission")
    axis = metric.tensor.coordinate_axis
    if scalar.variables != axis:
        raise OperationDomainValidationError(
            location=("scalar", "variables"),
            code="differential_geometry.laplace_beltrami.axis_mismatch",
            message="metric and scalar must use the same coordinate axis",
        )
    _recognize_source(metric, scalar, deadline)
    plan = build_plan(metric, scalar)
    request_checkpoint("after Laplace--Beltrami admission")
    (value,), determinant_guards = evaluate_admitted_dag(
        plan.dag.nodes,
        axis,
        fractions=(plan.fraction,),
        determinants=tuple(dict.fromkeys(plan.determinant.numerator)),
        sources=(),
        deadline=deadline,
        owner="Laplace--Beltrami",
        singular_metric=_singular_metric,
        noncanonical_location=("laplace_beltrami",),
        noncanonical_code="differential_geometry.laplace_beltrami.noncanonical_source",
        noncanonical_message="metric and scalar must be reduced canonical rational functions",
    )
    guards = canonical_locus_guards(
        metric.tensor.retained_nonzero_denominators,
        tuple(determinant_guards),
        component_denominators=(value.denominator, scalar.denominator),
        variable_count=len(axis),
    )
    return RationalLaplaceBeltramiResult._from_kernel(
        metric=metric,
        scalar=scalar,
        value=value,
        retained_nonzero_denominators=guards,
    )


__all__ = ["laplace_beltrami"]
