"""Exact rational metric pullback operation."""

from __future__ import annotations

import time

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.differential._recognition_process import (
    RationalFunctionRecognitionCandidate,
    recognize_canonical_rational_functions,
)
from jacobian.math.geometry.differential.metrics._dag_evaluate_process import (
    evaluate_admitted_dag,
)
from jacobian.math.geometry.differential.metrics._models import RationalCoordinateMetric
from jacobian.math.geometry.differential.pullback._models import (
    RationalMetricPullbackProfile,
)
from jacobian.math.geometry.differential.pullback._plan import build_plan
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    canonical_locus_guards,
)
from jacobian.math.polynomials._conversions import (
    sparse_rational_polynomial_from_sympy,
    sparse_rational_polynomial_to_sympy,
)
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_VARIABLES,
    RationalFunction,
    require_canonical_rational_function,
)


def _recognize_sources(
    components: tuple[RationalFunction, ...], deadline: float
) -> None:
    candidates = []
    for index, component in enumerate(dict.fromkeys(components)):
        request_checkpoint("before pullback source recognition")
        # Monomial denominators have an exact exponent-wise recognition path;
        # general polynomial GCDs run in the existing killable worker.
        if (
            not component.numerator.terms
            or not component.variables
            or len(component.denominator.terms) == 1
        ):
            try:
                require_canonical_rational_function(component)
            except PydanticCustomError as exc:
                raise OperationDomainValidationError(
                    location=("metric", "map"),
                    code="differential_geometry.rational_metric.pullback.noncanonical_source",
                    message="metric and map components must be reduced canonical rational functions",
                ) from exc
        else:
            candidates.append(
                RationalFunctionRecognitionCandidate(
                    owner="tensor", component=index, value=component
                )
            )
    recognition = recognize_canonical_rational_functions(
        tuple(candidates), deadline=deadline
    )
    if recognition.non_coprime is not None:
        raise OperationDomainValidationError(
            location=("metric", "map"),
            code="differential_geometry.rational_metric.pullback.noncanonical_source",
            message="metric and map components must be reduced canonical rational functions",
        )


def pullback_metric(
    metric: RationalCoordinateMetric, map_value: RationalFunctionMap
) -> RationalMetricPullbackProfile:
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return pullback_metric(metric, map_value)
    deadline = execution.started_at + 120
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("before rational metric pullback admission")
    try:
        metric = RationalCoordinateMetric.model_validate(
            metric.model_dump(mode="python")
        )
        map_value = RationalFunctionMap.model_validate(
            map_value.model_dump(mode="python")
        )
    except ValidationError as exc:
        raise OperationDomainValidationError(
            location=("metric", "map"),
            code="differential_geometry.rational_metric.pullback.invalid_source",
            message="metric and map must be canonical native values before planning",
        ) from exc
    if map_value.target_coordinates != metric.tensor.coordinate_axis:
        raise OperationDomainValidationError(
            location=("map", "target_coordinates"),
            code="differential_geometry.rational_metric.pullback.axis_mismatch",
            message="map target coordinates must equal metric coordinate axis",
        )
    n = len(map_value.source_variables)
    if not 1 <= n <= MAX_POLYNOMIAL_VARIABLES:
        raise OperationDomainValidationError(
            location=("map", "source_variables"),
            code="differential_geometry.rational_metric.pullback.axis",
            message=(
                "source coordinate axis must have between 1 and "
                f"{MAX_POLYNOMIAL_VARIABLES} coordinates"
            ),
        )
    plan = build_plan(metric, map_value)
    request_checkpoint("after complete rational metric pullback admission")
    _recognize_sources((*metric.tensor.components, *map_value.components), deadline)
    if any(not guard.scalar for guard in plan.guards if guard != plan.determinant):
        raise OperationDomainValidationError(
            location=("metric",),
            code="differential_geometry.rational_metric.pullback.undefined_metric_locus",
            message="a required metric denominator or chart guard vanishes identically after substitution",
        )
    if not plan.determinant.scalar:
        raise OperationDomainValidationError(
            location=("metric",),
            code="differential_geometry.rational_metric.pullback.singular_metric",
            message="metric determinant vanishes identically after substitution",
        )
    axis = map_value.source_variables
    unique_values = tuple(dict.fromkeys((*plan.output, *plan.guards)))
    components, _determinant_guards = evaluate_admitted_dag(
        plan.dag.nodes,
        axis,
        fractions=tuple(plan.fractions[value] for value in unique_values),
        determinants=tuple(dict.fromkeys(plan.determinant.numerator)),
        sources=(),
        deadline=deadline,
        owner="rational metric pullback",
        singular_metric=lambda: OperationDomainValidationError(
            location=("metric",),
            code="differential_geometry.rational_metric.pullback.singular_metric",
            message="metric determinant vanishes identically after substitution",
        ),
        undefined_metric_locus=lambda: OperationDomainValidationError(
            location=("metric",),
            code="differential_geometry.rational_metric.pullback.undefined_metric_locus",
            message="a required metric denominator or chart guard vanishes identically after substitution",
        ),
        noncanonical_location=("metric", "map"),
        noncanonical_code="differential_geometry.rational_metric.pullback.noncanonical_source",
        noncanonical_message="metric and map components must be reduced canonical rational functions",
    )
    normalized = dict(zip(unique_values, components, strict=True))
    for value in plan.guards:
        if not normalized[value].numerator.terms:
            if value == plan.determinant:
                raise OperationDomainValidationError(
                    location=("metric",),
                    code="differential_geometry.rational_metric.pullback.singular_metric",
                    message="metric determinant vanishes identically after substitution",
                )
            raise OperationDomainValidationError(
                location=("metric",),
                code="differential_geometry.rational_metric.pullback.undefined_metric_locus",
                message="a required metric denominator or chart guard vanishes identically after substitution",
            )
    guards = tuple(
        sparse_rational_polynomial_from_sympy(
            sparse_rational_polynomial_to_sympy(
                normalized[value].numerator, axis
            ).monic(),
            axis,
            maximum_terms=256,
        )
        for value in plan.guards
        if normalized[value].numerator.terms
    )
    output = tuple(normalized[value] for value in plan.output)
    guard_polynomials = tuple(guards)
    tensor = RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=("COVARIANT", "COVARIANT"),
        components=output,
        retained_nonzero_denominators=canonical_locus_guards(
            guard_polynomials,
            component_denominators=tuple(value.denominator for value in output),
            variable_count=n,
        ),
    )
    profile = RationalMetricPullbackProfile(
        metric=metric,
        map=map_value,
        pullback=tensor,
        pullback_locus_guard=tensor.retained_nonzero_denominators,
    )
    request_checkpoint("after rational metric pullback construction")
    return profile


__all__ = ["pullback_metric"]
