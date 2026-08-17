"""Domain adapter for polynomial map operations backed by SymPy."""

from __future__ import annotations

import sympy
from sympy import Symbol, simplify, sympify

from jacobian.contracts.polynomial_maps import (
    CompositionRequest,
    CompositionResult,
    EvalRequest,
    EvalResult,
    JacobianRequest,
    JacobianResult,
)


def evaluate_polynomial(request: EvalRequest) -> EvalResult:
    """Evaluate a polynomial at a rational point using SymPy."""
    expr = sympify(request.polynomial.expression)
    substitutions = {}
    for var_name, var_value in zip(
        request.point.variables,
        request.point.values,
        strict=True,
    ):
        sym = Symbol(var_name)
        substitutions[sym] = var_value.as_fraction()

    result = simplify(expr.subs(substitutions))
    if result.free_symbols:
        raise ValueError("evaluation point must cover every free variable")
    return EvalResult(value=str(result))


def compute_jacobian(request: JacobianRequest) -> JacobianResult:
    """Compute the Jacobian matrix of a polynomial map using SymPy."""
    input_vars = [Symbol(v) for v in request.input_variables]
    outputs = [sympify(p.expression) for p in request.output_polynomials]

    n_outputs = len(outputs)
    n_inputs = len(input_vars)

    entries = []
    for i in range(n_outputs):
        for j in range(n_inputs):
            derivative = sympy.diff(outputs[i], input_vars[j])
            entries.append(str(derivative))

    return JacobianResult(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        entries=tuple(entries),
    )


def compose_polynomials(request: CompositionRequest) -> CompositionResult:
    """Compose outer(inner(x)) using SymPy substitution."""
    outer_expr = sympify(request.outer.expression)
    inner_expr = sympify(request.inner.expression)
    outer_var = Symbol(request.outer_variable)

    composed = outer_expr.subs(outer_var, inner_expr)
    composed = simplify(composed)

    return CompositionResult(expression=str(composed))


__all__ = ["compose_polynomials", "compute_jacobian", "evaluate_polynomial"]
