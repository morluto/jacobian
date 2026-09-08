"""Execute admitted exact Laplace--Beltrami plans."""

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
from jacobian.math.geometry.differential.laplace_beltrami._models import (
    RationalLaplaceBeltramiResult,
)
from jacobian.math.geometry.differential.laplace_beltrami._plan import build_plan
from jacobian.math.geometry.differential.metrics._models import RationalCoordinateMetric
from jacobian.math.geometry.differential.metrics._normalize_process import (
    cancel_fraction,
)
from jacobian.math.geometry.differential.metrics._plan import singular
from jacobian.math.geometry.differential.metrics.operations import _evaluate_node
from jacobian.math.geometry.differential.values import canonical_locus_guards
from jacobian.math.polynomials._conversions import (
    sparse_rational_polynomial_from_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    require_canonical_rational_function,
)


def _recognize_source(
    metric: RationalCoordinateMetric, scalar: RationalFunction
) -> None:
    for source in (*metric.tensor.components, scalar):
        request_checkpoint("before Laplace--Beltrami source recognition")
        try:
            require_canonical_rational_function(source)
        except PydanticCustomError as exc:
            raise OperationDomainValidationError(
                location=("laplace_beltrami",),
                code="differential_geometry.laplace_beltrami.noncanonical_source",
                message="metric and scalar must be reduced canonical rational functions",
            ) from exc


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
    plan = build_plan(metric, scalar)
    request_checkpoint("after Laplace--Beltrami admission")
    _recognize_source(metric, scalar)
    from sympy import QQ, Poly

    symbols = symbols_for_variables(axis)
    cache: dict[int, Any] = {
        0: Poly(0, *symbols, domain=QQ),
        1: Poly(1, *symbols, domain=QQ),
    }
    numerator, denominator = plan.fraction
    raw_numerator = _evaluate_node(numerator, plan.dag.nodes, axis, cache)
    raw_denominator = _evaluate_node(denominator, plan.dag.nodes, axis, cache)
    if raw_denominator.is_zero:
        raise singular()
    cancelled_numerator, cancelled_denominator = cancel_fraction(
        raw_numerator, raw_denominator, deadline=deadline
    )
    value = RationalFunction._from_kernel(
        variables=axis,
        numerator=sparse_rational_polynomial_from_sympy(
            cancelled_numerator, axis, maximum_terms=256
        ),
        denominator=sparse_rational_polynomial_from_sympy(
            cancelled_denominator, axis, maximum_terms=256
        ),
    )
    determinant_guards = []
    for index in dict.fromkeys(plan.determinant.numerator):
        polynomial = _evaluate_node(index, plan.dag.nodes, axis, cache)
        if polynomial.is_zero:
            raise singular()
        determinant_guards.append(
            sparse_rational_polynomial_from_sympy(
                polynomial.monic(), axis, maximum_terms=256
            )
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
