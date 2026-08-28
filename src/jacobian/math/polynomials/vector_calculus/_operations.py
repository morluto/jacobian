"""Domain functions for polynomial vector calculus operations."""

from __future__ import annotations

from collections.abc import Iterable

import sympy

from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import RationalPolynomial
from jacobian.math.polynomials.vector_calculus._models import (
    CurlRequest,
    DirectionalDerivativeRequest,
    ScalarFieldRequest,
    ScalarResult,
    VectorFieldRequest,
    VectorResult,
)


def _wire(expression: sympy.Expr, variables: tuple[str, ...]) -> RationalPolynomial:
    return rational_polynomial_from_sympy(
        sympy.Poly(
            sympy.expand(expression), *symbols_for_variables(variables), domain=sympy.QQ
        ),
        variables,
        maximum_terms=256,
    )


def _expressions(
    polynomials: Iterable[RationalPolynomial],
) -> tuple[sympy.Expr, ...]:
    return tuple(rational_polynomial_to_sympy(item).as_expr() for item in polynomials)


def compute_gradient(request: ScalarFieldRequest) -> VectorResult:
    variables = request.polynomial.variables
    expression = rational_polynomial_to_sympy(request.polynomial).as_expr()
    return VectorResult(
        components=tuple(
            _wire(sympy.diff(expression, variable), variables)
            for variable in symbols_for_variables(variables)
        ),
        method="SYMPY_GRADIENT",
    )


def compute_divergence(request: VectorFieldRequest) -> ScalarResult:
    variables = request.components[0].variables
    expression = sum(
        sympy.diff(component, variable)
        for component, variable in zip(
            _expressions(request.components),
            symbols_for_variables(variables),
            strict=True,
        )
    )
    return ScalarResult(
        result=_wire(expression, variables),
        method="SYMPY_DIVERGENCE",
    )


def compute_curl(request: CurlRequest) -> VectorResult:
    """Return the standard three-dimensional curl of a polynomial field."""

    variables = request.components[0].variables
    x, y, z = symbols_for_variables(variables)
    fx, fy, fz = _expressions(request.components)
    return VectorResult(
        components=(
            _wire(sympy.diff(fz, y) - sympy.diff(fy, z), variables),
            _wire(sympy.diff(fx, z) - sympy.diff(fz, x), variables),
            _wire(sympy.diff(fy, x) - sympy.diff(fx, y), variables),
        ),
        method="SYMPY_CURL",
    )


def compute_laplacian(request: ScalarFieldRequest) -> ScalarResult:
    variables = request.polynomial.variables
    expression = rational_polynomial_to_sympy(request.polynomial).as_expr()
    laplacian = sum(
        sympy.diff(expression, variable, 2)
        for variable in symbols_for_variables(variables)
    )
    return ScalarResult(
        result=_wire(laplacian, variables),
        method="SYMPY_LAPLACIAN",
    )


def compute_directional_derivative(
    request: DirectionalDerivativeRequest,
) -> ScalarResult:
    variables = request.polynomial.variables
    expression = rational_polynomial_to_sympy(request.polynomial).as_expr()
    gradient = (
        sympy.diff(expression, variable)
        for variable in symbols_for_variables(variables)
    )
    direction = (
        sympy.Rational(*coordinate.as_integer_ratio())
        for coordinate in request.direction
    )
    return ScalarResult(
        result=_wire(
            sum(
                derivative * coordinate
                for derivative, coordinate in zip(gradient, direction, strict=True)
            ),
            variables,
        ),
        method="SYMPY_DIRECTIONAL_DERIVATIVE",
    )


__all__ = [
    "compute_curl",
    "compute_directional_derivative",
    "compute_divergence",
    "compute_gradient",
    "compute_laplacian",
]
