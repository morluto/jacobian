"""Domain functions for polynomial vector calculus operations."""

from __future__ import annotations

import sympy

from jacobian.math.polynomial_vector_calc._models import (
    DirectionalDerivativeRequest,
    ScalarFieldRequest,
    ScalarResult,
    VectorFieldRequest,
    VectorResult,
)


def _parse_poly(expr_str: str, variables: tuple[str, ...]) -> sympy.Expr:
    """Parse a polynomial expression string with given variable names.

    Validates that the result is a polynomial in exactly the declared variables
    with rational coefficients, rejecting transcendental expressions, foreign
    symbols, and non-polynomial inputs.
    """
    var_symbols = sympy.symbols(variables)
    if isinstance(var_symbols, sympy.Symbol):
        var_symbols = (var_symbols,)
    try:
        expr = sympy.sympify(
            expr_str, locals=dict(zip(variables, var_symbols, strict=True))
        )
    except Exception as exc:
        raise ValueError(f"failed to parse expression: {expr_str}") from exc
    # is_polynomial returns True, False, or None (unknown); treat None as False
    if expr.is_polynomial(*var_symbols) is not True:
        raise ValueError(
            f"expression must be a polynomial in {variables}, "
            f"got non-polynomial: {expr}"
        )
    free_symbols = expr.free_symbols
    allowed = set(var_symbols)
    extra = free_symbols - allowed
    if extra:
        raise ValueError(
            f"expression contains undeclared symbols: {sorted(s.name for s in extra)}"
        )
    return expr


def compute_gradient(request: ScalarFieldRequest) -> VectorResult:
    var_symbols = sympy.symbols(request.variables)
    if isinstance(var_symbols, sympy.Symbol):
        var_symbols = (var_symbols,)
    poly = _parse_poly(request.polynomial, request.variables)
    grads = [sympy.diff(poly, v) for v in var_symbols]
    return VectorResult(
        components=tuple(str(g) for g in grads),
        variables=request.variables,
        method="SYMPY_GRADIENT",
    )


def compute_divergence(request: VectorFieldRequest) -> ScalarResult:
    var_symbols = sympy.symbols(request.variables)
    if isinstance(var_symbols, sympy.Symbol):
        var_symbols = (var_symbols,)
    polys = [_parse_poly(c, request.variables) for c in request.components]
    div = sum(sympy.diff(p, v) for p, v in zip(polys, var_symbols, strict=True))
    return ScalarResult(
        result=str(sympy.expand(div)),
        variables=request.variables,
        method="SYMPY_DIVERGENCE",
    )


def compute_curl(request: VectorFieldRequest) -> VectorResult:
    """Curl in 3D: (d_z F_y - d_y F_z, d_x F_z - d_z F_x, d_y F_x - d_x F_y)."""
    if len(request.variables) != 3:
        raise ValueError("curl is defined for 3D vector fields")
    x, y, z = sympy.symbols(request.variables)
    fx, fy, fz = (
        _parse_poly(request.components[0], request.variables),
        _parse_poly(request.components[1], request.variables),
        _parse_poly(request.components[2], request.variables),
    )
    curl_x = sympy.diff(fz, y) - sympy.diff(fy, z)
    curl_y = sympy.diff(fx, z) - sympy.diff(fz, x)
    curl_z = sympy.diff(fy, x) - sympy.diff(fx, y)
    return VectorResult(
        components=(
            str(sympy.expand(curl_x)),
            str(sympy.expand(curl_y)),
            str(sympy.expand(curl_z)),
        ),
        variables=request.variables,
        method="SYMPY_CURL",
    )


def compute_laplacian(request: ScalarFieldRequest) -> ScalarResult:
    var_symbols = sympy.symbols(request.variables)
    if isinstance(var_symbols, sympy.Symbol):
        var_symbols = (var_symbols,)
    poly = _parse_poly(request.polynomial, request.variables)
    laplacian = sum(sympy.diff(poly, v, 2) for v in var_symbols)
    return ScalarResult(
        result=str(sympy.expand(laplacian)),
        variables=request.variables,
        method="SYMPY_LAPLACIAN",
    )


def compute_directional_derivative(
    request: DirectionalDerivativeRequest,
) -> ScalarResult:
    var_symbols = sympy.symbols(request.variables)
    if isinstance(var_symbols, sympy.Symbol):
        var_symbols = (var_symbols,)
    poly = _parse_poly(request.polynomial, request.variables)
    grad = [sympy.diff(poly, v) for v in var_symbols]
    direction = [sympy.sympify(d) for d in request.direction]
    result = sum(g * d for g, d in zip(grad, direction, strict=True))
    return ScalarResult(
        result=str(sympy.expand(result)),
        variables=request.variables,
        method="SYMPY_DIRECTIONAL_DERIVATIVE",
    )
