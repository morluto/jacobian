"""Exact pullback of rational coordinate metrics along rational maps."""

from __future__ import annotations

import time
from typing import Any

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
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
    rational_function_from_sympy,
    rational_function_to_sympy,
    sparse_rational_polynomial_from_sympy,
    sparse_rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.rational_functions.composition.operations import (
    compose_maps,
)
from jacobian.math.polynomials.rational_functions.gradient._kernel import (
    _normalize_fraction,
)
from jacobian.math.polynomials.rational_functions.maps.operations import jacobian_matrix
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _unit(variables: tuple[str, ...]) -> SparseRationalPolynomial:
    return SparseRationalPolynomial(
        terms=(
            RationalPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=1),
                exponents=(0,) * len(variables),
            ),
        )
    )


def _as_function(
    polynomial: SparseRationalPolynomial, variables: tuple[str, ...]
) -> RationalFunction:
    return RationalFunction(
        variables=variables, numerator=polynomial, denominator=_unit(variables)
    )


def _synthetic_map(
    variables: tuple[str, ...], components: tuple[RationalFunction, ...], prefix: str
) -> RationalFunctionMap:
    return RationalFunctionMap(
        source_variables=variables,
        target_coordinates=tuple(f"{prefix}{i}" for i in range(len(components))),
        components=components,
    )


def _metric_substitution(
    metric: RationalCoordinateMetric, map_value: RationalFunctionMap
) -> Any:
    axis = metric.tensor.coordinate_axis
    components = metric.tensor.components
    outer = _synthetic_map(axis, components, "g")
    return compose_maps(outer, map_value)


def _determinant_substitution(
    metric: RationalCoordinateMetric, map_value: RationalFunctionMap
) -> Any:
    from sympy import Matrix

    axis = metric.tensor.coordinate_axis
    matrix = Matrix(
        [
            [
                rational_function_to_sympy(metric.tensor.components[i * len(axis) + j])
                for j in range(len(axis))
            ]
            for i in range(len(axis))
        ]
    )
    determinant = rational_function_from_sympy(matrix.det(), axis)
    return compose_maps(_synthetic_map(axis, (determinant,), "det"), map_value)


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
    build_plan(metric, map_value)
    substituted = _metric_substitution(metric, map_value)
    determinant = _determinant_substitution(metric, map_value)
    if not determinant.composite.components[0].numerator.terms:
        raise OperationDomainValidationError(
            location=("metric", "tensor"),
            code="differential_geometry.rational_metric.pullback.singular_metric",
            message="metric determinant vanishes identically after substitution",
        )
    jacobian = jacobian_matrix(map_value)
    axis = metric.tensor.coordinate_axis
    xaxis = map_value.source_variables
    xsymbols = symbols_for_variables(xaxis)
    substituted_values = tuple(
        rational_function_to_sympy(value) for value in substituted.composite.components
    )
    entries = jacobian.entries
    metric_size = len(metric.tensor.coordinate_axis)
    output = []
    for a in range(len(xaxis)):
        for b in range(len(xaxis)):
            request_checkpoint("during rational metric pullback contraction")
            expression = None
            for i in range(metric_size):
                for j in range(metric_size):
                    left = rational_function_to_sympy(entries[i][a])
                    middle = substituted_values[i * metric_size + j]
                    right = rational_function_to_sympy(entries[j][b])
                    term = left * middle * right
                    expression = term if expression is None else expression + term
            if expression is None:
                expression = 0
            from sympy import Poly, together

            numerator, denominator = together(expression).as_numer_denom()
            output.append(
                _normalize_fraction(
                    Poly(numerator, *xsymbols, domain="QQ"),
                    Poly(denominator, *xsymbols, domain="QQ"),
                    xaxis,
                )
            )
    guards = [guard.polynomial for guard in substituted.construction_locus_guard]
    guards.extend(guard.polynomial for guard in determinant.construction_locus_guard)
    determinant_value = determinant.composite.components[0]
    if any(term.exponents for term in determinant_value.numerator.terms):
        guards.append(
            sparse_rational_polynomial_from_sympy(
                sparse_rational_polynomial_to_sympy(
                    determinant_value.numerator, xaxis
                ).monic(),
                xaxis,
            )
        )
    for guard in metric.tensor.retained_nonzero_denominators:
        guard_map = _synthetic_map(axis, (_as_function(guard, axis),), "h")
        guard_result = compose_maps(guard_map, map_value)
        guard_value = guard_result.composite.components[0]
        if not guard_value.numerator.terms:
            raise OperationDomainValidationError(
                location=("metric", "retained_nonzero_denominators"),
                code="differential_geometry.rational_metric.pullback.undefined_metric_locus",
                message="a retained metric locus guard vanishes identically after substitution",
            )
        guards.append(
            sparse_rational_polynomial_from_sympy(
                sparse_rational_polynomial_to_sympy(
                    guard_value.numerator, xaxis
                ).monic(),
                xaxis,
            )
        )
    guard_polynomials = tuple(guards)
    result_tensor = RationalCoordinateTensor(
        coordinate_axis=xaxis,
        variance=("COVARIANT", "COVARIANT"),
        components=tuple(output),
        retained_nonzero_denominators=canonical_locus_guards(
            guard_polynomials,
            component_denominators=tuple(value.denominator for value in output),
            variable_count=len(xaxis),
        ),
    )
    return RationalMetricPullbackProfile(
        metric=metric,
        map=map_value,
        pullback=result_tensor,
        pullback_locus_guard=result_tensor.retained_nonzero_denominators,
    )


__all__ = ["pullback_metric"]
