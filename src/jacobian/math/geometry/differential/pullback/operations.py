"""Exact rational metric pullback operation."""

from __future__ import annotations

import time
from typing import Any

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
from jacobian.math.geometry.differential.metrics._dag import Expression
from jacobian.math.geometry.differential.metrics._models import RationalCoordinateMetric
from jacobian.math.geometry.differential.metrics._normalize_process import (
    cancel_fraction,
)
from jacobian.math.geometry.differential.metrics.operations import _evaluate_node
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
    symbols_for_variables,
)
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap
from jacobian.math.polynomials.values import (
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
    if map_value.target_coordinates != metric.tensor.coordinate_axis:
        raise OperationDomainValidationError(
            location=("map", "target_coordinates"),
            code="differential_geometry.rational_metric.pullback.axis_mismatch",
            message="map target coordinates must equal metric coordinate axis",
        )
    n = len(map_value.source_variables)
    if not 1 <= n <= 4:
        raise OperationDomainValidationError(
            location=("map", "source_variables"),
            code="differential_geometry.rational_metric.pullback.axis",
            message="source coordinate axis must have between 1 and 4 coordinates",
        )
    plan = build_plan(metric, map_value)
    request_checkpoint("after complete rational metric pullback admission")
    _recognize_sources((*metric.tensor.components, *map_value.components), deadline)
    from sympy import QQ, Poly

    axis = map_value.source_variables
    symbols = symbols_for_variables(axis)
    cache = {0: Poly(0, *symbols, domain=QQ), 1: Poly(1, *symbols, domain=QQ)}

    def raw(value: Expression) -> tuple[Any, Any]:
        numerator, denominator = plan.fractions[value]
        return (
            _evaluate_node(
                numerator,
                plan.dag.nodes,
                axis,
                cache,
            ),
            _evaluate_node(denominator, plan.dag.nodes, axis, cache),
        )

    normalized: dict[Expression, tuple[Any, Any]] = {}

    def normalize(value: Expression) -> tuple[Any, Any]:
        if value not in normalized:
            request_checkpoint("before rational metric pullback normalization")
            numerator, denominator = raw(value)
            normalized[value] = cancel_fraction(
                numerator, denominator, deadline=deadline
            )
            request_checkpoint("after rational metric pullback normalization")
        return normalized[value]

    guards = []
    for guard in plan.guards:
        numerator, _ = raw(guard)
        if numerator.is_zero:
            if guard == plan.determinant:
                raise OperationDomainValidationError(
                    location=("metric",),
                    code="differential_geometry.rational_metric.pullback.singular_metric",
                    message="metric determinant vanishes identically after substitution",
                )
            raise OperationDomainValidationError(
                location=("metric",),
                code="differential_geometry.rational_metric.pullback.undefined_metric_locus",
                message="a required metric or map denominator guard vanishes identically after substitution",
            )
        normalized_numerator, _ = normalize(guard)
        guards.append(
            sparse_rational_polynomial_from_sympy(
                normalized_numerator.monic(), axis, maximum_terms=256
            )
        )
    functions: dict[Expression, RationalFunction] = {}

    def convert(value: Expression) -> RationalFunction:
        if value not in functions:
            num, den = normalize(value)
            functions[value] = RationalFunction._from_kernel(
                variables=axis,
                numerator=sparse_rational_polynomial_from_sympy(
                    num, axis, maximum_terms=256
                ),
                denominator=sparse_rational_polynomial_from_sympy(
                    den, axis, maximum_terms=256
                ),
            )
        return functions[value]

    output = tuple(convert(value) for value in plan.output)
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
