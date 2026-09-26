"""Exact native operations for plane algebraic curves."""

from __future__ import annotations

from dataclasses import dataclass

import sympy

from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.algebraic_curves._arclength import enclose_arclength
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
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)


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


@dataclass(frozen=True)
class PlaneCurveBlowupChartData:
    """Canonical strict-transform and exceptional-divisor data for one chart."""

    exceptional_multiplicity: int
    strict_transform: RationalPolynomial
    exceptional_intersection_polynomial: RationalPolynomial


def _digit_floor(value: int) -> int:
    return decimal_digit_width(value) - 1


def _blowup_result_admission_error(location: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=(location,),
        code="plane_algebraic_curve.blowup_coefficient_growth_over_envelope",
        message="the exact chart substitution exceeded the 128-digit curve coefficient bound",
    )


def plane_curve_blowup_chart(
    polynomial: RationalPolynomial,
    center: VariablePoint,
    radial_variable: PolynomialVariable,
    slope_variable: PolynomialVariable,
) -> PlaneCurveBlowupChartData:
    """Return the strict transform in x=a+u, y=b+u*t and its E intersection."""
    _admit_curve_polynomial(polynomial)
    if type(center) is not VariablePoint:
        _domain_error(
            "blowup_center_invalid",
            "the blowup center must be a canonical rational point",
            "center",
        )
    try:
        center = VariablePoint.model_validate(center.model_dump(), strict=True)
    except (TypeError, ValueError, AttributeError) as exc:
        raise OperationDomainValidationError(
            location=("center",),
            code="plane_algebraic_curve.blowup_center_invalid",
            message="the blowup center must satisfy its canonical carrier shape",
        ) from exc
    if len(polynomial.variables) != 2:
        _domain_error(
            "blowup_axis_invalid",
            "a plane-curve blowup requires exactly two source variables",
            "polynomial",
        )
    if radial_variable == slope_variable:
        _domain_error(
            "blowup_chart_axis_collision",
            "chart variables must be distinct",
            "radial_variable",
        )
    if {radial_variable, slope_variable} & set(polynomial.variables):
        _domain_error(
            "blowup_chart_axis_collision",
            "chart variables must be fresh relative to the source axes",
            "radial_variable",
        )
    if center.variables != polynomial.variables:
        _domain_error(
            "blowup_center_axis_invalid",
            "the center must use the complete ordered source axis",
            "center",
        )
    source = rational_polynomial_to_sympy(polynomial)
    if source.is_zero or source.total_degree() < 1:
        _domain_error(
            "blowup_curve_invalid",
            "the blowup source must be a nonzero curve polynomial",
            "polynomial",
        )
    x, y = symbols_for_variables(polynomial.variables)
    radial, slope = symbols_for_variables((radial_variable, slope_variable))
    a, b = (value.as_fraction() for value in center.values)

    # Bound the full binomial expansion, output carrier, and coefficient
    # growth before asking SymPy to expand the translated chart substitution.
    # Translated terms combine over the least common multiple of the distinct
    # source denominators multiplied by center-coordinate numerator and
    # denominator powers up to the curve degree, so the bound accumulates
    # that joint height instead of trusting the largest single coefficient.
    terms = polynomial.polynomial.terms
    raw_terms = sum((term.exponents[0] + 1) * (term.exponents[1] + 1) for term in terms)
    if raw_terms > 1_000_000 or raw_terms > 256:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="plane_algebraic_curve.blowup_expansion_over_envelope",
            message="the predicted strict-transform expansion exceeds its 256-term bound",
        )
    degree = int(source.total_degree())
    source_numerators = [term.coefficient.as_fraction().numerator for term in terms]
    source_denominators = {term.coefficient.as_fraction().denominator for term in terms}
    growth_bound = (
        1
        + sum(_digit_floor(denominator) for denominator in source_denominators)
        + max(_digit_floor(numerator) for numerator in source_numerators)
        + degree
        * (
            _digit_floor(a.numerator)
            + _digit_floor(a.denominator)
            + _digit_floor(b.numerator)
            + _digit_floor(b.denominator)
            + 1
        )
        + _digit_floor(raw_terms)
    )
    if growth_bound > 128:
        raise OperationResourceAdmissionError(
            location=("center",),
            code="plane_algebraic_curve.blowup_coefficient_growth_over_envelope",
            message="the exact chart substitution may exceed the 128-digit curve coefficient bound",
        )

    source_expression = source.as_expr()
    if (
        source_expression.subs(
            {
                x: sympy.Rational(a.numerator, a.denominator),
                y: sympy.Rational(b.numerator, b.denominator),
            }
        )
        != 0
    ):
        _domain_error(
            "blowup_center_not_on_curve",
            "the blowup center must lie on the source curve",
            "center",
        )

    pulled = sympy.Poly(
        sympy.expand(
            source_expression.subs(
                {
                    x: sympy.Rational(a.numerator, a.denominator) + radial,
                    y: sympy.Rational(b.numerator, b.denominator) + radial * slope,
                },
                simultaneous=True,
            )
        ),
        radial,
        slope,
        domain=sympy.QQ,
    )
    if pulled.is_zero:
        _domain_error(
            "blowup_pullback_zero", "the chart pullback must be nonzero", "polynomial"
        )
    multiplicity = min(exponents[0] for exponents, _ in pulled.terms())
    strict_poly = sympy.Poly(
        {
            (exponents[0] - multiplicity, exponents[1]): coefficient
            for exponents, coefficient in pulled.terms()
        },
        radial,
        slope,
        domain=sympy.QQ,
    )
    exceptional_poly = sympy.Poly(
        strict_poly.as_expr().subs(radial, 0), slope, domain=sympy.QQ
    )
    if exceptional_poly.is_zero:
        _domain_error(
            "blowup_exceptional_restriction_zero",
            "the exceptional restriction must be nonzero after removing the exact multiplicity",
            "polynomial",
        )
    try:
        strict_transform = rational_polynomial_from_sympy(
            strict_poly,
            (radial_variable, slope_variable),
            maximum_terms=256,
        )
        require_polynomial_budget(
            strict_transform,
            maximum_terms=256,
            maximum_exponent=128,
            maximum_coefficient_digits=128,
            label="strict-transform polynomial",
        )
    except ValueError as exc:
        raise _blowup_result_admission_error("strict_transform") from exc
    if any(sum(term.exponents) > 128 for term in strict_transform.polynomial.terms):
        raise OperationResourceAdmissionError(
            location=("strict_transform",),
            code="plane_algebraic_curve.blowup_result_degree_over_envelope",
            message="the strict-transform polynomial exceeds the 128-degree output bound",
        )
    exceptional_poly = exceptional_poly.monic()
    try:
        exceptional_intersection = rational_polynomial_from_sympy(
            exceptional_poly, (slope_variable,), maximum_terms=256
        )
        require_polynomial_budget(
            exceptional_intersection,
            maximum_terms=256,
            maximum_exponent=64,
            maximum_coefficient_digits=128,
            label="exceptional intersection polynomial",
        )
    except ValueError as exc:
        raise _blowup_result_admission_error("exceptional_intersection") from exc
    replay = sympy.Poly(
        strict_poly.as_expr() * radial**multiplicity, radial, slope, domain=sympy.QQ
    )
    if replay != pulled:
        _domain_error(
            "blowup_identity_failed",
            "the strict-transform divisibility identity failed",
            "strict_transform",
        )
    return PlaneCurveBlowupChartData(
        exceptional_multiplicity=multiplicity,
        strict_transform=strict_transform,
        exceptional_intersection_polynomial=exceptional_intersection,
    )


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
    except OperationResourceAdmissionError:
        raise
    except (OperationDomainValidationError, ValueError, TypeError):
        return False


__all__ = [
    "affine_chart",
    "affine_curve_check",
    "enclose_arclength",
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
