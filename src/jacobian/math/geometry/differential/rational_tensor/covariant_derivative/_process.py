"""Killable SymPy expansion and cancellation for covariant derivatives."""

from __future__ import annotations

from typing import Any

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.differential.metrics._dag_process import (
    DagDegeneracy,
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
    *,
    deadline: float,
) -> tuple[tuple[RationalFunction, ...], tuple[Any, ...]]:
    """Expand and cancel the admitted DAG in one killable worker."""

    evaluation = evaluate_admitted_dag(
        plan.dag.nodes,
        axis,
        fractions=tuple(plan.fractions[value] for value in plan.derivative),
        determinants=tuple(dict.fromkeys(plan.determinant.numerator)),
        deadline=deadline,
        owner="covariant derivative",
    )
    if isinstance(evaluation, DagDegeneracy):
        raise _singular_metric()
    return evaluation.fractions, evaluation.determinants
