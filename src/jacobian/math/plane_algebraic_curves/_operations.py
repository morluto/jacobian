"""Domain functions for plane algebraic curve operations."""

from __future__ import annotations

import sympy

from jacobian.math.plane_algebraic_curves._conic import (
    derive_rational_conic_parametrization,
)
from jacobian.math.plane_algebraic_curves._models import (
    HOMOGENIZING_COORDINATE,
    AffineChartRequest,
    AffineChartResult,
    AffineCurveRequest,
    AffineCurveResult,
    ProjectiveClosureRequest,
    ProjectiveClosureResult,
    RationalConicParametrizationRequest,
    RationalConicParametrizationResult,
)
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)


def compute_affine_curve_check(request: AffineCurveRequest) -> AffineCurveResult:
    """Check whether a nonconstant polynomial defines an affine plane curve."""

    polynomial = rational_polynomial_to_sympy(request.polynomial)
    degree = 0 if polynomial.is_zero else int(polynomial.total_degree())
    return AffineCurveResult(
        is_valid=not polynomial.is_zero and degree >= 1,
        degree=degree,
    )


def compute_projective_closure(
    request: ProjectiveClosureRequest,
) -> ProjectiveClosureResult:
    """Homogenize an affine plane curve with the reserved coordinate ``z``."""

    source = rational_polynomial_to_sympy(request.polynomial)
    source_variables = symbols_for_variables(request.polynomial.variables)
    homogenizing = sympy.Symbol(HOMOGENIZING_COORDINATE)
    variables = (*request.polynomial.variables, HOMOGENIZING_COORDINATE)
    degree = 0 if source.is_zero else int(source.total_degree())
    expression = sum(
        coefficient
        * sympy.prod(
            variable**exponent
            for variable, exponent in zip(
                source_variables,
                monomial,
                strict=True,
            )
        )
        * homogenizing ** (degree - sum(monomial))
        for monomial, coefficient in source.terms()
    )
    closure = sympy.Poly(
        sympy.expand(expression),
        *source_variables,
        homogenizing,
        domain=sympy.QQ,
    )
    return ProjectiveClosureResult(
        polynomial=rational_polynomial_from_sympy(
            closure,
            variables,
            maximum_terms=256,
        )
    )


def compute_affine_chart(request: AffineChartRequest) -> AffineChartResult:
    """Dehomogenize a projective plane curve at one chart coordinate."""

    source = rational_polynomial_to_sympy(request.polynomial)
    chart_index = request.polynomial.variables.index(request.chart_variable)
    symbols = symbols_for_variables(request.polynomial.variables)
    remaining_variables = tuple(
        variable
        for index, variable in enumerate(request.polynomial.variables)
        if index != chart_index
    )
    remaining_symbols = tuple(
        symbol for index, symbol in enumerate(symbols) if index != chart_index
    )
    chart = sympy.Poly(
        sympy.expand(source.as_expr().subs(symbols[chart_index], 1)),
        *remaining_symbols,
        domain=sympy.QQ,
    )
    return AffineChartResult(
        polynomial=rational_polynomial_from_sympy(
            chart,
            remaining_variables,
            maximum_terms=256,
        )
    )


def compute_rational_conic_parametrization(
    request: RationalConicParametrizationRequest,
) -> RationalConicParametrizationResult:
    """Parametrize a smooth rational conic by its normalized line pencil."""

    data = derive_rational_conic_parametrization(
        request.polynomial,
        request.point,
        request.parameter,
    )
    return RationalConicParametrizationResult._from_kernel(
        request,
        coordinates=data.coordinates,
        inverse_parameter=data.inverse_parameter,
        finite_parameter_denominator=data.finite_parameter_denominator,
    )


__all__ = [
    "compute_affine_chart",
    "compute_affine_curve_check",
    "compute_projective_closure",
    "compute_rational_conic_parametrization",
]
