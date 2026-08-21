"""Domain-owned polynomial map operations backed by SymPy."""

from __future__ import annotations

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
    JacobianRequest,
    JacobianResult,
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


def compute_jacobian(request: JacobianRequest) -> JacobianResult:
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


__all__ = ["compose_polynomials", "compute_jacobian", "evaluate_polynomial"]
