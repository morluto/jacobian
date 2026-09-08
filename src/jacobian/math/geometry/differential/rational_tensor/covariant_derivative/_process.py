"""Killable SymPy expansion and cancellation for covariant derivatives."""

from __future__ import annotations

from typing import Any

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.differential.metrics._dag_process import (
    RationalDagWorkerMessages,
    evaluate_admitted_rational_dag,
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


_MESSAGES = RationalDagWorkerMessages(
    timeout_before="covariant derivative deadline expired before DAG expansion",
    timeout_after="covariant derivative deadline expired after payload encoding",
    timeout_during="covariant derivative deadline expired during polynomial DAG expansion",
    cancelled_during="covariant derivative cancelled during polynomial DAG expansion",
    start_failure="bounded covariant-derivative worker could not be started",
    malformed="bounded covariant-derivative worker returned malformed output",
    directory_prefix="jacobian-covariant-derivative-",
    checkpoint_prefix="covariant-derivative",
    noncanonical_location=("covariant_derivative",),
    noncanonical_code="differential_geometry.covariant_derivative.noncanonical_source",
    noncanonical_message=(
        "metric and tensor components must be reduced canonical rational functions"
    ),
    singular_metric=_singular_metric,
)


def evaluate_admitted_covariant_derivative(
    plan: Plan,
    axis: tuple[str, ...],
    sources: tuple[RationalFunction, ...],
    *,
    deadline: float,
) -> tuple[tuple[RationalFunction, ...], tuple[Any, ...]]:
    """Expand and cancel the admitted DAG in one killable worker."""

    return evaluate_admitted_rational_dag(
        plan.dag.nodes,
        [list(plan.fractions[value]) for value in plan.derivative],
        list(dict.fromkeys(plan.determinant.numerator)),
        axis,
        sources,
        deadline=deadline,
        messages=_MESSAGES,
    )
