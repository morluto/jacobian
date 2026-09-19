"""Exact rational metric pullback operation."""

from __future__ import annotations

import time
from collections.abc import Sized

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
from jacobian.math.geometry.differential.metrics._dag_process import (
    DagDegeneracy,
    evaluate_admitted_dag,
)
from jacobian.math.geometry.differential.metrics._models import RationalCoordinateMetric
from jacobian.math.geometry.differential.pullback._models import (
    RationalMetricPullbackProfile,
)
from jacobian.math.geometry.differential.pullback._plan import build_plan
from jacobian.math.geometry.differential.values import (
    MAX_RATIONAL_TENSOR_COMPONENTS,
    MAX_RATIONAL_TENSOR_LOCUS_GUARDS,
    RationalCoordinateTensor,
    canonical_locus_guards,
)
from jacobian.math.polynomials._conversions import (
    sparse_rational_polynomial_from_sympy,
    sparse_rational_polynomial_to_sympy,
)
from jacobian.math.polynomials.rational_functions.values import (
    MAX_RATIONAL_MAP_COMPONENTS,
    RationalFunctionMap,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_VARIABLES,
    RationalFunction,
    SparseRationalPolynomial,
    require_canonical_rational_function,
)


def _bounded_sized(
    value: object,
    limit: int,
    *,
    location: tuple[str, ...],
    message: str,
) -> None:
    if not isinstance(value, Sized) or isinstance(value, (str, bytes, bytearray)):
        raise OperationDomainValidationError(
            location=location,
            code="differential_geometry.rational_metric.pullback.invalid_source",
            message=message,
        )
    if len(value) > limit:
        raise OperationDomainValidationError(
            location=location,
            code="differential_geometry.rational_metric.pullback.invalid_source",
            message=message,
        )


def _preflight_pullback_sources(
    metric: object, map_value: object
) -> tuple[RationalCoordinateMetric, RationalFunctionMap]:
    if type(metric) is not RationalCoordinateMetric:
        raise OperationDomainValidationError(
            location=("metric",),
            code="differential_geometry.rational_metric.pullback.invalid_source",
            message="metric must be a RationalCoordinateMetric",
        )
    if type(map_value) is not RationalFunctionMap:
        raise OperationDomainValidationError(
            location=("map",),
            code="differential_geometry.rational_metric.pullback.invalid_source",
            message="map must be a RationalFunctionMap",
        )
    tensor = getattr(metric, "tensor", None)
    axis = getattr(tensor, "coordinate_axis", None)
    components = getattr(tensor, "components", None)
    _bounded_sized(
        axis,
        4,
        location=("metric", "tensor", "coordinate_axis"),
        message="metric coordinate axis exceeds the admitted envelope",
    )
    _bounded_sized(
        components,
        MAX_RATIONAL_TENSOR_COMPONENTS,
        location=("metric", "tensor", "components"),
        message="metric tensor exceeds the admitted component envelope",
    )
    if isinstance(axis, Sized) and isinstance(components, Sized):
        expected = len(axis) ** 2
        if len(components) != expected:
            raise OperationDomainValidationError(
                location=("metric", "tensor", "components"),
                code="differential_geometry.rational_metric.pullback.invalid_source",
                message="metric and map must be canonical native values before planning",
            )
    _bounded_sized(
        getattr(tensor, "retained_nonzero_denominators", None),
        MAX_RATIONAL_TENSOR_LOCUS_GUARDS,
        location=("metric", "tensor", "retained_nonzero_denominators"),
        message="metric locus guards exceed the admitted envelope",
    )
    _bounded_sized(
        getattr(map_value, "components", None),
        MAX_RATIONAL_MAP_COMPONENTS,
        location=("map", "components"),
        message="map exceeds the admitted component envelope",
    )
    _bounded_sized(
        getattr(map_value, "source_variables", None),
        MAX_POLYNOMIAL_VARIABLES,
        location=("map", "source_variables"),
        message="map source axis exceeds the admitted envelope",
    )
    targets = getattr(map_value, "target_coordinates", None)
    components = getattr(map_value, "components", None)
    if (
        isinstance(targets, Sized)
        and isinstance(components, Sized)
        and len(components) != len(targets)
    ):
        raise OperationDomainValidationError(
            location=("map", "components"),
            code="differential_geometry.rational_metric.pullback.invalid_source",
            message="metric and map must be canonical native values before planning",
        )
    try:
        metric = RationalCoordinateMetric.model_validate(metric.model_dump())
        map_value = RationalFunctionMap.model_validate(map_value.model_dump())
    except ValidationError as exc:
        raise OperationDomainValidationError(
            location=("metric", "map"),
            code="differential_geometry.rational_metric.pullback.invalid_source",
            message="metric and map must be canonical native values before planning",
        ) from exc
    return metric, map_value


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
    metric, map_value = _preflight_pullback_sources(metric, map_value)
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
    if any(not guard.scalar for guard in plan.guards[:-1]):
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
    evaluation = evaluate_admitted_dag(
        plan.dag.nodes,
        axis,
        fractions=tuple(plan.fractions[value] for value in unique_values),
        determinants=tuple(dict.fromkeys(plan.determinant.numerator)),
        undefined_numerators=tuple(
            dict.fromkeys(plan.fractions[value][0] for value in plan.guards[:-1])
        ),
        deadline=deadline,
        owner="rational metric pullback",
    )
    if isinstance(evaluation, DagDegeneracy):
        if evaluation.kind == "undefined":
            raise OperationDomainValidationError(
                location=("metric",),
                code="differential_geometry.rational_metric.pullback.undefined_metric_locus",
                message="a required metric denominator or chart guard vanishes identically after substitution",
            )
        raise OperationDomainValidationError(
            location=("metric",),
            code="differential_geometry.rational_metric.pullback.singular_metric",
            message="metric determinant vanishes identically after substitution",
        )
    components = evaluation.fractions
    normalized = dict(zip(unique_values, components, strict=True))
    for value in plan.guards[:-1]:
        if not normalized[value].numerator.terms:
            raise OperationDomainValidationError(
                location=("metric",),
                code="differential_geometry.rational_metric.pullback.undefined_metric_locus",
                message="a required metric denominator or chart guard vanishes identically after substitution",
            )
    if not normalized[plan.determinant].numerator.terms:
        raise OperationDomainValidationError(
            location=("metric",),
            code="differential_geometry.rational_metric.pullback.singular_metric",
            message="metric determinant vanishes identically after substitution",
        )
    guarded: list[SparseRationalPolynomial] = []
    for guard_index, value in enumerate(plan.guards):
        if guard_index % 64 == 0:
            request_checkpoint("during pullback guard normalization")
        numerator = normalized[value].numerator
        if not numerator.terms:
            continue
        guarded.append(
            sparse_rational_polynomial_from_sympy(
                sparse_rational_polynomial_to_sympy(numerator, axis).monic(),
                axis,
                maximum_terms=256,
            )
        )
    guards = tuple(guarded)
    request_checkpoint("after pullback guard normalization")
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
