"""Domain-owned polynomial map operations backed by SymPy."""

from __future__ import annotations

import time

import sympy

from jacobian.math.polynomials._conversions import (
    rational_from_sympy,
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.maps._models import (
    CompositionRequest,
    CompositionResult,
    EvalRequest,
    EvalResult,
    GenericDegreeOutcome,
    GenericDegreeRequest,
    GenericDegreeResult,
    JacobianResult,
)
from jacobian.math.polynomials.maps._replay import (
    run_bounded_certificate_replay,
)
from jacobian.math.polynomials.maps._singular import run_singular_generic_fiber
from jacobian.math.polynomials.maps.values import RationalPolynomialMap

_REPLAY_FAILURE_OUTCOMES: dict[str, tuple[GenericDegreeOutcome, str]] = {
    "CANCELLED": (
        "CANCELLED",
        "Certificate replay was cancelled before producing a result.",
    ),
    "TIMEOUT": ("TIMEOUT", "Certificate replay exceeded the declared wall-time limit."),
    "LIMIT_EXCEEDED": (
        "BOUND_EXCEEDED",
        "The generic-fiber certificate replay exceeded the declared computation bound.",
    ),
    "INVALID": ("ERROR", "Singular evidence failed source-bound exact replay."),
    "ERROR": ("ERROR", "The certificate replay worker failed."),
}


def compute_generic_degree(request: GenericDegreeRequest) -> GenericDegreeResult:
    """Compute the exact degree of the map's generic scheme-theoretic fiber."""

    deadline = time.monotonic() + request.resource_budget.wall_seconds
    backend = run_singular_generic_fiber(
        request.polynomial_map,
        request.resource_budget,
    )
    if backend.outcome != "COMPUTED":
        operational_outcome = (
            "BOUND_EXCEEDED" if backend.outcome == "LIMIT_EXCEEDED" else backend.outcome
        )
        return GenericDegreeResult(
            outcome=operational_outcome,
            source=request.polynomial_map,
            detail=backend.detail,
        )
    if backend.certificate is None or backend.dimension is None:
        return GenericDegreeResult(
            outcome="ERROR",
            source=request.polynomial_map,
            detail="Singular returned incomplete generic-fiber evidence.",
        )
    mathematical_outcome: GenericDegreeOutcome
    if backend.dimension == -1:
        mathematical_outcome = "NOT_DOMINANT"
        degree = None
    elif backend.dimension == 0:
        if backend.vector_dimension is None:
            return GenericDegreeResult(
                outcome="ERROR",
                source=request.polynomial_map,
                detail="Singular returned a finite fiber without its exact degree.",
            )
        mathematical_outcome = "GENERICALLY_FINITE"
        degree = backend.vector_dimension
    else:
        mathematical_outcome = "DOMINANT_NOT_GENERICALLY_FINITE"
        degree = None
    remaining_seconds = deadline - time.monotonic()
    if remaining_seconds <= 0:
        return GenericDegreeResult(
            outcome="TIMEOUT",
            source=request.polynomial_map,
            detail=(
                "The declared wall-time envelope expired before certificate replay."
            ),
        )
    replay = run_bounded_certificate_replay(
        request.polynomial_map,
        backend.certificate,
        wall_seconds=remaining_seconds,
    )
    if replay.status != "COMPUTED":
        outcome, default_detail = _REPLAY_FAILURE_OUTCOMES[replay.status]
        return GenericDegreeResult(
            outcome=outcome,
            source=request.polynomial_map,
            detail=replay.detail or default_detail,
        )
    if replay.outcome != mathematical_outcome or replay.degree != degree:
        return GenericDegreeResult(
            outcome="ERROR",
            source=request.polynomial_map,
            detail=("Singular metadata disagree with the exact certificate replay."),
        )
    return GenericDegreeResult._from_kernel(
        outcome=mathematical_outcome,
        source=request.polynomial_map,
        degree=degree,
        evidence=backend.certificate,
        detail=None,
    )


def evaluate_polynomial(request: EvalRequest) -> EvalResult:
    """Evaluate one exact polynomial at its complete ordered rational point."""

    polynomial = rational_polynomial_to_sympy(request.polynomial)
    substitutions = dict(
        zip(
            symbols_for_variables(request.point.variables),
            (value.as_fraction() for value in request.point.values),
            strict=True,
        )
    )
    value = polynomial.as_expr().subs(substitutions)
    return EvalResult(value=rational_from_sympy(value))


def compute_jacobian(request: RationalPolynomialMap) -> JacobianResult:
    """Compute a row-major Jacobian over the map's source ring."""

    variables = symbols_for_variables(request.input_variables)
    outputs = [
        rational_polynomial_to_sympy(polynomial).as_expr()
        for polynomial in request.output_polynomials
    ]
    entries = tuple(
        rational_polynomial_from_sympy(
            sympy.Poly(sympy.diff(output, variable), *variables, domain=sympy.QQ),
            request.input_variables,
            maximum_terms=256,
        )
        for output in outputs
        for variable in variables
    )
    return JacobianResult(
        n_inputs=len(variables),
        n_outputs=len(outputs),
        entries=entries,
    )


def compose_polynomials(request: CompositionRequest) -> CompositionResult:
    """Substitute the inner univariate polynomial into the outer polynomial."""

    outer = rational_polynomial_to_sympy(request.outer).as_expr()
    inner = rational_polynomial_to_sympy(request.inner).as_expr()
    outer_variable = symbols_for_variables(request.outer.variables)[0]
    inner_variable = symbols_for_variables(request.inner.variables)[0]
    composition = sympy.Poly(
        sympy.expand(outer.subs(outer_variable, inner)),
        inner_variable,
        domain=sympy.QQ,
    )
    return CompositionResult(
        polynomial=rational_polynomial_from_sympy(
            composition,
            request.inner.variables,
            maximum_terms=256,
        )
    )


__all__ = [
    "compose_polynomials",
    "compute_generic_degree",
    "compute_jacobian",
    "evaluate_polynomial",
]
