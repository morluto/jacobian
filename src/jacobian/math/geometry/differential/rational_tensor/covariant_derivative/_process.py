"""Killable SymPy expansion and cancellation for covariant derivatives."""

from __future__ import annotations

from typing import Any

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.differential.metrics._dag_evaluate_process import (
    evaluate_admitted_dag,
)
from jacobian.math.geometry.differential.rational_tensor.covariant_derivative._plan import (
    Plan,
)
from jacobian.math.polynomials.values import RationalFunction


def _singular_metric() -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("metric",),
        code="differential_geometry.covariant_derivative.singular_metric",
        message="metric determinant is identically zero",
    )


def evaluate_admitted_covariant_derivative(
    plan: Plan,
    axis: tuple[str, ...],
    sources: tuple[RationalFunction, ...],
    *,
    deadline: float,
) -> tuple[tuple[RationalFunction, ...], tuple[Any, ...]]:
    """Expand and cancel the admitted DAG in one killable worker."""

    return evaluate_admitted_dag(
        plan.dag.nodes,
        axis,
        fractions=tuple(plan.fractions[value] for value in plan.derivative),
        determinants=tuple(dict.fromkeys(plan.determinant.numerator)),
        sources=sources,
        deadline=deadline,
        owner="covariant derivative",
        singular_metric=_singular_metric,
        noncanonical_location=("covariant_derivative",),
        noncanonical_code="differential_geometry.covariant_derivative.noncanonical_source",
        noncanonical_message=(
            "metric and tensor components must be reduced canonical rational functions"
        ),
    )
