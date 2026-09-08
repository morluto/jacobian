"""Execute admitted exact Levi-Civita covariant-derivative plans."""

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
from jacobian.math.geometry.differential.metrics._dag import Expression
from jacobian.math.geometry.differential.metrics._models import (
    RationalCoordinateMetric,
)
from jacobian.math.geometry.differential.metrics._plan import singular
from jacobian.math.geometry.differential.metrics.operations import _evaluate_node
from jacobian.math.geometry.differential.rational_tensor.covariant_derivative._models import (
    RationalCovariantDerivativeProfile,
)
from jacobian.math.geometry.differential.rational_tensor.covariant_derivative._plan import (
    Plan,
    build_plan,
)
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    canonical_locus_guards,
)
from jacobian.math.polynomials._conversions import (
    sparse_rational_polynomial_from_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.rational_functions.gradient._kernel import (
    _normalize_fraction,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
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


def _determinant_guards(
    plan: Plan, axis: tuple[str, ...], cache: dict[int, Any]
) -> tuple[Any, ...]:
    guards = []
    for index in dict.fromkeys(plan.determinant.numerator):
        polynomial = _evaluate_node(index, plan.dag.nodes, axis, cache)
        if polynomial.is_zero:
            raise singular()
        guards.append(
            sparse_rational_polynomial_from_sympy(
                polynomial.monic(), axis, maximum_terms=256
            )
        )
    return tuple(guards)


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

    from sympy import QQ, Poly

    axis = metric.tensor.coordinate_axis
    symbols = symbols_for_variables(axis)
    cache: dict[int, Any] = {
        0: Poly(0, *symbols, domain=QQ),
        1: Poly(1, *symbols, domain=QQ),
    }

    normalized: dict[Expression, RationalFunction] = {}

    def convert(value: Expression) -> RationalFunction:
        request_checkpoint("before covariant-derivative component normalization")
        if value not in normalized:
            numerator, denominator = plan.fractions[value]
            raw_numerator = _evaluate_node(numerator, plan.dag.nodes, axis, cache)
            raw_denominator = _evaluate_node(denominator, plan.dag.nodes, axis, cache)
            normalized[value] = _normalize_fraction(
                raw_numerator, raw_denominator, axis
            )
        request_checkpoint("after covariant-derivative component normalization")
        return normalized[value]

    determinant_guards = _determinant_guards(plan, axis, cache)
    components = tuple(convert(value) for value in plan.derivative)
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
