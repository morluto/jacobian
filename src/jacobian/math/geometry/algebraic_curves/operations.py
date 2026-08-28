"""Exact native operations for plane algebraic curves."""

from __future__ import annotations

import sympy

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.algebraic_curves._conic import (
    ConicParametrizationData,
    derive_rational_conic_parametrization,
    validate_rational_conic_request,
)
from jacobian.math.geometry.algebraic_curves._models import (
    _MAX_CURVE_TERMS,
    HOMOGENIZING_COORDINATE,
    _require_curve_polynomial,
    _validation_error,
    _validation_error_from,
)
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.maps._models import VariablePoint
from jacobian.math.polynomials.values import PolynomialVariable, RationalPolynomial


def _domain_error(reason: str, message: str, *location: str) -> None:
    error = _validation_error(reason, message)
    raise OperationDomainValidationError(
        location=location, code=error.type, message=error.message()
    )


def _admit_curve_polynomial(polynomial: RationalPolynomial) -> None:
    try:
        _require_curve_polynomial(polynomial)
    except ValueError as exc:
        classified = _validation_error_from(exc)
        raise OperationDomainValidationError(
            location=("polynomial",),
            code=classified.type,
            message=classified.message(),
        ) from exc


def affine_curve_check(polynomial: RationalPolynomial) -> tuple[bool, int]:
    """Return whether a polynomial defines an affine plane curve and its degree."""
    _admit_curve_polynomial(polynomial)
    if len(polynomial.variables) != 2:
        _domain_error(
            "affine_axis_invalid",
            "affine plane curves require exactly two variables",
            "polynomial",
        )
    source = rational_polynomial_to_sympy(polynomial)
    degree = 0 if source.is_zero else int(source.total_degree())
    return (not source.is_zero and degree >= 1, degree)


def projective_closure(polynomial: RationalPolynomial) -> RationalPolynomial:
    """Homogenize an affine plane curve with the reserved coordinate ``z``."""
    _admit_curve_polynomial(polynomial)
    if len(polynomial.variables) != 2:
        _domain_error(
            "closure_axis_invalid",
            "projective closure requires exactly two variables",
            "polynomial",
        )
    if HOMOGENIZING_COORDINATE in polynomial.variables:
        _domain_error(
            "homogenizing_coordinate_reserved",
            "affine variable axis must not contain the reserved "
            f"homogenizing coordinate {HOMOGENIZING_COORDINATE!r}",
            "polynomial",
        )
    source = rational_polynomial_to_sympy(polynomial)
    source_variables = symbols_for_variables(polynomial.variables)
    homogenizing = sympy.Symbol(HOMOGENIZING_COORDINATE)
    variables = (*polynomial.variables, HOMOGENIZING_COORDINATE)
    degree = 0 if source.is_zero else int(source.total_degree())
    expression = sum(
        coefficient
        * sympy.prod(
            variable**exponent
            for variable, exponent in zip(source_variables, monomial, strict=True)
        )
        * homogenizing ** (degree - sum(monomial))
        for monomial, coefficient in source.terms()
    )
    closure = sympy.Poly(
        sympy.expand(expression), *source_variables, homogenizing, domain=sympy.QQ
    )
    return rational_polynomial_from_sympy(
        closure, variables, maximum_terms=_MAX_CURVE_TERMS
    )


def affine_chart(
    polynomial: RationalPolynomial, chart_variable: PolynomialVariable
) -> RationalPolynomial:
    """Dehomogenize a projective plane curve at one chart coordinate."""
    _admit_curve_polynomial(polynomial)
    if len(polynomial.variables) != 3:
        _domain_error(
            "chart_axis_invalid",
            "projective plane curves require exactly three variables",
            "polynomial",
        )
    if chart_variable not in polynomial.variables:
        _domain_error(
            "chart_variable_axis_mismatch",
            "chart_variable must belong to the polynomial axis",
            "chart_variable",
        )
    source = rational_polynomial_to_sympy(polynomial)
    if not source.is_homogeneous:
        _domain_error(
            "polynomial_not_homogeneous",
            "projective polynomial must be homogeneous",
            "polynomial",
        )
    chart_index = polynomial.variables.index(chart_variable)
    symbols = symbols_for_variables(polynomial.variables)
    remaining_variables = tuple(
        variable
        for index, variable in enumerate(polynomial.variables)
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
    return rational_polynomial_from_sympy(
        chart, remaining_variables, maximum_terms=_MAX_CURVE_TERMS
    )


def rational_conic_parametrization(
    polynomial: RationalPolynomial,
    point: VariablePoint,
    parameter: PolynomialVariable,
) -> ConicParametrizationData:
    """Parametrize a smooth rational conic by its normalized line pencil."""
    try:
        validate_rational_conic_request(polynomial, point, parameter)
    except ValueError as exc:
        classified = _validation_error_from(exc)
        raise OperationDomainValidationError(
            location=("request",),
            code=classified.type,
            message=classified.message(),
        ) from exc
    return derive_rational_conic_parametrization(polynomial, point, parameter)


__all__ = [
    "affine_chart",
    "affine_curve_check",
    "projective_closure",
    "rational_conic_parametrization",
]
