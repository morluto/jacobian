"""Domain functions for plane algebraic curve operations."""

from __future__ import annotations

import sympy

from jacobian.math.plane_algebraic_curves._models import (
    AffineChartRequest,
    AffineChartResult,
    AffineCurveRequest,
    AffineCurveResult,
    ProjectiveClosureRequest,
    ProjectiveClosureResult,
    _parse_polynomial,
)

HOMOGENIZING_COORDINATE = "z"


def compute_affine_curve_check(request: AffineCurveRequest) -> AffineCurveResult:
    """Check that a polynomial defines a valid affine plane curve."""
    poly = _parse_polynomial(request.polynomial, request.variables)
    degree = int(sympy.total_degree(poly))
    is_valid = poly != 0 and degree >= 1
    return AffineCurveResult(
        is_valid=is_valid,
        degree=degree,
    )


def compute_projective_closure(
    request: ProjectiveClosureRequest,
) -> ProjectiveClosureResult:
    """Compute the projective closure by homogenizing with a new variable."""
    var_symbols = list(sympy.symbols(request.variables))
    poly = _parse_polynomial(request.polynomial, request.variables)
    z = sympy.Symbol(HOMOGENIZING_COORDINATE)
    terms = sympy.Poly(poly, *var_symbols)
    degree = terms.total_degree()
    new_terms = []
    for monom, coeff in terms.as_dict().items():
        total_deg = sum(monom)
        factor = z ** (degree - total_deg)
        term = coeff
        for i, exp in enumerate(monom):
            term *= var_symbols[i] ** exp
        new_terms.append(term * factor)
    homogenized = sympy.expand(sum(new_terms))
    return ProjectiveClosureResult(
        polynomial=str(homogenized),
        variables=(*request.variables, HOMOGENIZING_COORDINATE),
    )


def compute_affine_chart(request: AffineChartRequest) -> AffineChartResult:
    """Extract an affine chart by dehomogenizing at the chart variable."""
    chart_var = sympy.Symbol(request.chart_variable)
    poly = _parse_polynomial(request.polynomial, request.variables)
    idx = request.variables.index(request.chart_variable)
    other_vars = [v for i, v in enumerate(request.variables) if i != idx]

    dehomogenized = poly.subs(chart_var, 1)
    dehomogenized = sympy.expand(dehomogenized)

    return AffineChartResult(
        polynomial=str(dehomogenized),
        variables=tuple(other_vars),
    )
