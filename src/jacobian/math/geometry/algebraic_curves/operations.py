"""Exact native operations for plane algebraic curves."""

from __future__ import annotations

import sympy

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.algebraic_curves._conic import (
    ConicParametrizationData,
    derive_rational_conic_parametrization,
    validate_rational_conic_request,
)
from jacobian.math.geometry.algebraic_curves._gaussian_realification import (
    GaussianRealificationResult,
    UnivariateGaussianPolynomial,
)
from jacobian.math.geometry.algebraic_curves._gaussian_realification import (
    gaussian_realification as _gaussian_realification,
)
from jacobian.math.geometry.algebraic_curves._gaussian_realification import (
    verify_gaussian_realification as _verify_gaussian_realification,
)
from jacobian.math.geometry.algebraic_curves._models import (
    _MAX_CURVE_TERMS,
    HOMOGENIZING_COORDINATE,
    AffineChartResult,
    AffineCurveResult,
    ProjectiveClosureResult,
    RationalConicParametrizationResult,
    _require_curve_polynomial,
    _validation_error,
    _validation_error_from,
)
from jacobian.math.geometry.algebraic_curves._singularity import singularity_profile
from jacobian.math.geometry.algebraic_curves._singularity_models import (
    ProjectivePlaneCurveSingularityProfile,
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


def gaussian_realification(
    polynomial: UnivariateGaussianPolynomial,
    target_variables: tuple[PolynomialVariable, PolynomialVariable],
) -> GaussianRealificationResult:
    """Return the real and imaginary parts after substituting ``x + i*y``."""
    return _gaussian_realification(polynomial, target_variables)


def verify_gaussian_realification(claim: GaussianRealificationResult) -> bool:
    """Verify real and imaginary components against the retained source."""

    return _verify_gaussian_realification(claim)


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


def verify_affine_curve_check(claim: AffineCurveResult) -> bool:
    """Check the affine-curve decision asserted by a serialized claim."""

    try:
        return affine_curve_check(claim.polynomial) == (
            claim.is_valid,
            claim.degree,
        )
    except (OperationDomainValidationError, ValueError, TypeError):
        return False


def verify_projective_closure(claim: ProjectiveClosureResult) -> bool:
    """Verify a retained affine source and its exact homogenization."""

    try:
        return projective_closure(claim.source_polynomial) == claim.polynomial
    except (OperationDomainValidationError, ValueError, TypeError):
        return False


def verify_affine_chart(claim: AffineChartResult) -> bool:
    """Verify a retained projective source, chart, and dehomogenization."""

    try:
        return (
            affine_chart(claim.source_polynomial, claim.chart_variable)
            == claim.polynomial
        )
    except (OperationDomainValidationError, ValueError, TypeError):
        return False


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


def verify_rational_conic_parametrization(
    claim: RationalConicParametrizationResult,
) -> bool:
    """Verify the source-bound line-pencil identities of a conic claim."""

    try:
        data = rational_conic_parametrization(
            claim.source_polynomial,
            claim.exceptional_point,
            claim.parameter,
        )
        return (
            data.coordinates == claim.coordinates
            and data.inverse_parameter == claim.inverse_parameter
            and data.finite_parameter_denominator == claim.finite_parameter_denominator
        )
    except (OperationDomainValidationError, ValueError, TypeError):
        return False


def verify_projective_plane_curve_singularity_profile(
    claim: ProjectivePlaneCurveSingularityProfile,
) -> bool:
    """Verify a complete singularity profile against its retained source."""

    try:
        return singularity_profile(claim.source_polynomial) == claim
    except (OperationDomainValidationError, ValueError, TypeError, RuntimeError):
        return False


__all__ = [
    "affine_chart",
    "affine_curve_check",
    "projective_closure",
    "rational_conic_parametrization",
    "singularity_profile",
    "verify_affine_chart",
    "verify_affine_curve_check",
    "verify_gaussian_realification",
    "verify_projective_closure",
    "verify_projective_plane_curve_singularity_profile",
    "verify_rational_conic_parametrization",
]
