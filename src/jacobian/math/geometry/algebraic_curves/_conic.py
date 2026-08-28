"""Exact kernel and admission helpers for smooth rational conics."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm
from typing import Any

import sympy

from jacobian._exact import require_bounded_rational
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials.maps._models import VariablePoint
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalFunction,
    RationalPolynomial,
    require_polynomial_budget,
)

MAX_CONIC_TERMS = 6
MAX_CONIC_INPUT_DIGITS = 128
MAX_CONIC_INTERMEDIATE_DIGITS = 16_384
MAX_CONIC_RESULT_DIGITS = 128
MAX_CONIC_RESULT_TERMS = 3

# A factor of a primitive integer polynomial of degree at most two has
# coefficient norm at most 2**2 times the source 2-norm. With at most three
# coefficients, one extra decimal digit bounds 4*sqrt(3).
_QUADRATIC_FACTOR_DIGIT_SLACK = 1
_LOG10_2_UPPER_NUMERATOR = 30_103
_LOG10_2_UPPER_DENOMINATOR = 100_000


@dataclass(frozen=True, slots=True)
class ConicParametrizationData:
    """Canonical values produced by the gradient-normalized line pencil."""

    coordinates: tuple[RationalFunction, RationalFunction]
    inverse_parameter: RationalFunction
    finite_parameter_denominator: RationalPolynomial


@dataclass(frozen=True, slots=True)
class _IntegerPolynomialHeight:
    primitive_coefficient_digits: int
    cleared_coefficient_digits: int
    rational_content: Fraction


@dataclass(frozen=True, slots=True)
class _ParametrizationHeightBounds:
    intermediate_digits: int
    result_coefficient_digits: int


def _coefficient(polynomial: Any, exponents: tuple[int, int]) -> Any:
    return polynomial.coeff_monomial(exponents)


def _point_values(point: VariablePoint) -> tuple[Any, Any]:
    first, second = point.values
    return (
        sympy.Rational(*first.as_integer_ratio()),
        sympy.Rational(*second.as_integer_ratio()),
    )


def _source_data(
    polynomial: RationalPolynomial,
    point: VariablePoint,
) -> tuple[Any, tuple[Any, Any], tuple[Any, Any], tuple[Any, Any, Any]]:
    source = rational_polynomial_to_sympy(polynomial)
    point_values = _point_values(point)
    substitutions = dict(zip(source.gens, point_values, strict=True))
    gradient = tuple(
        source.diff(variable).eval(substitutions) for variable in source.gens
    )
    quadratic = (
        _coefficient(source, (2, 0)),
        _coefficient(source, (1, 1)),
        _coefficient(source, (0, 2)),
    )
    return source, point_values, gradient, quadratic


def _source_coefficients(
    polynomial: RationalPolynomial,
) -> dict[tuple[int, ...], Fraction]:
    return {
        term.exponents: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }


def _digits(value: int) -> int:
    bit_length = abs(value).bit_length()
    if bit_length == 0:
        return 1
    return (
        bit_length * _LOG10_2_UPPER_NUMERATOR + _LOG10_2_UPPER_DENOMINATOR - 1
    ) // _LOG10_2_UPPER_DENOMINATOR


def _rational_digits(value: Fraction) -> int:
    return max(_digits(value.numerator), _digits(value.denominator))


def _integer_polynomial_height(
    coefficients: tuple[Fraction, ...],
) -> _IntegerPolynomialHeight:
    """Measure the primitive integer associate after bounded denominator clearing."""

    nonzero = tuple(coefficient for coefficient in coefficients if coefficient)
    if not nonzero:
        raise ValueError("parametrization polynomial cannot be zero")
    clearing_denominator = lcm(*(coefficient.denominator for coefficient in nonzero))
    integer_coefficients = tuple(
        coefficient.numerator * (clearing_denominator // coefficient.denominator)
        for coefficient in nonzero
    )
    content = gcd(*(abs(coefficient) for coefficient in integer_coefficients))
    primitive_coefficients = tuple(
        coefficient // content for coefficient in integer_coefficients
    )
    return _IntegerPolynomialHeight(
        primitive_coefficient_digits=max(map(_digits, primitive_coefficients)),
        cleared_coefficient_digits=max(map(_digits, integer_coefficients)),
        rational_content=Fraction(content, clearing_denominator),
    )


def _rational_function_height_bounds(
    numerator: tuple[Fraction, ...],
    denominator: tuple[Fraction, ...],
) -> tuple[int, int]:
    """Bound reduced coefficients and degree-two cancellation intermediates."""

    numerator_integer = _integer_polynomial_height(numerator)
    denominator_integer = _integer_polynomial_height(denominator)
    factor_slack = _QUADRATIC_FACTOR_DIGIT_SLACK
    content_ratio = numerator_integer.rational_content / (
        denominator_integer.rational_content
    )
    numerator_factor_digits = (
        numerator_integer.primitive_coefficient_digits + factor_slack
    )
    denominator_factor_digits = (
        denominator_integer.primitive_coefficient_digits + factor_slack
    )

    # Write each rational polynomial as rational content times a primitive
    # integer polynomial. Mignotte's degree-two factor bound controls the
    # quotients after gcd cancellation. Dividing by the remaining denominator
    # leading coefficient accounts for canonical monic normalization.
    result_digits = max(
        _digits(content_ratio.numerator) + numerator_factor_digits,
        _digits(content_ratio.denominator) + denominator_factor_digits,
        denominator_factor_digits,
    )

    # For degrees at most two, gcd subresultants are minors of a Sylvester
    # matrix of order at most four. Four coefficient products plus two decimal
    # digits for the at-most 4! determinant terms conservatively bound them.
    cancellation_digits = (
        4
        * max(
            numerator_integer.cleared_coefficient_digits,
            denominator_integer.cleared_coefficient_digits,
        )
        + 2
    )
    return result_digits, cancellation_digits


def _coprime_linear_function_height_bounds(
    numerator: tuple[Fraction, ...],
    denominator: tuple[Fraction, ...],
) -> tuple[int, int]:
    """Bound the inverse chart, whose two affine linear forms are coprime."""

    leading = next(coefficient for coefficient in denominator if coefficient)
    result_digits = max(
        _rational_digits(coefficient / leading)
        for coefficient in (*numerator, *denominator)
    )
    numerator_integer = _integer_polynomial_height(numerator)
    denominator_integer = _integer_polynomial_height(denominator)
    # The only possible nonconstant gcd of two affine linear forms is detected
    # by their 2x2 coefficient minors. Each minor has two products.
    cancellation_digits = (
        2
        * max(
            numerator_integer.cleared_coefficient_digits,
            denominator_integer.cleared_coefficient_digits,
        )
        + 1
    )
    return result_digits, cancellation_digits


def _monic_polynomial_result_digits(
    coefficients: tuple[Fraction, ...],
) -> int:
    leading = coefficients[-1]
    return max(_rational_digits(coefficient / leading) for coefficient in coefficients)


def _parametrization_height_bounds(
    polynomial: RationalPolynomial,
    point: VariablePoint,
) -> _ParametrizationHeightBounds:
    """Bound raw work and every normalized materialized coefficient."""

    coefficients = _source_coefficients(polynomial)
    zero = Fraction(0)
    a = coefficients.get((2, 0), zero)
    b = coefficients.get((1, 1), zero)
    c = coefficients.get((0, 2), zero)
    d = coefficients.get((1, 0), zero)
    e = coefficients.get((0, 1), zero)
    constant = coefficients.get((0, 0), zero)
    px, py = (value.as_fraction() for value in point.values)
    fx = 2 * a * px + b * py + d
    fy = b * px + 2 * c * py + e
    gradient_square = fx * fx + fy * fy

    denominator = (
        a * fx * fx + b * fx * fy + c * fy * fy,
        -2 * a * fx * fy + b * (fx * fx - fy * fy) + 2 * c * fx * fy,
        a * fy * fy - b * fx * fy + c * fx * fx,
    )
    coordinate_x_numerator = (
        px * denominator[0] - gradient_square * fx,
        px * denominator[1] + gradient_square * fy,
        px * denominator[2],
    )
    coordinate_y_numerator = (
        py * denominator[0] - gradient_square * fy,
        py * denominator[1] - gradient_square * fx,
        py * denominator[2],
    )
    inverse_numerator = (
        -fy,
        fx,
        fy * px - fx * py,
    )
    inverse_denominator = (
        fx,
        fy,
        -fx * px - fy * py,
    )

    rational_function_bounds = (
        _rational_function_height_bounds(coordinate_x_numerator, denominator),
        _rational_function_height_bounds(coordinate_y_numerator, denominator),
        _coprime_linear_function_height_bounds(inverse_numerator, inverse_denominator),
    )
    result_digits = max(
        *(result for result, _intermediate in rational_function_bounds),
        _monic_polynomial_result_digits(denominator),
    )

    determinant = (
        8 * a * c * constant
        + 2 * b * d * e
        - 2 * c * d * d
        - 2 * b * b * constant
        - 2 * a * e * e
    )
    raw_values = (
        fx,
        fy,
        gradient_square,
        determinant,
        *denominator,
        *coordinate_x_numerator,
        *coordinate_y_numerator,
        *inverse_numerator,
        *inverse_denominator,
    )
    # Canonical RationalFunction validation performs one more gcd on at most
    # three result coefficients. The same degree-two determinant bound is
    # 12*result_digits + 2 after clearing their denominators.
    intermediate_digits = max(
        *(_rational_digits(value) for value in raw_values),
        *(intermediate for _result, intermediate in rational_function_bounds),
        12 * result_digits + 2,
    )
    return _ParametrizationHeightBounds(
        intermediate_digits=intermediate_digits,
        result_coefficient_digits=result_digits,
    )


def _require_source_bounds(
    polynomial: RationalPolynomial,
    point: VariablePoint,
) -> None:
    require_polynomial_budget(
        polynomial,
        maximum_terms=MAX_CONIC_TERMS,
        maximum_exponent=2,
        maximum_coefficient_digits=MAX_CONIC_INPUT_DIGITS,
        label="conic polynomial",
    )
    for coordinate in point.values:
        require_bounded_rational(
            coordinate,
            max_digits=MAX_CONIC_INPUT_DIGITS,
            label="conic point coordinate",
        )


def _require_work_and_result_bounds(
    polynomial: RationalPolynomial,
    point: VariablePoint,
) -> None:
    bounds = _parametrization_height_bounds(polynomial, point)
    if bounds.result_coefficient_digits > MAX_CONIC_RESULT_DIGITS:
        raise ValueError(
            "conic parametrization normalized output exceeds the conservative "
            "128-digit coefficient bound"
        )
    if bounds.intermediate_digits > MAX_CONIC_INTERMEDIATE_DIGITS:
        raise ValueError(
            "conic parametrization exceeds the 16,384-digit intermediate bound"
        )


def _require_smooth_conic_and_point(
    polynomial: RationalPolynomial,
    point: VariablePoint,
) -> None:
    # Exact preflight on the already bounded canonical coefficients; the
    # backend is entered only after every admission check has passed.
    coefficients = _source_coefficients(polynomial)
    zero = Fraction(0)
    a = coefficients.get((2, 0), zero)
    b = coefficients.get((1, 1), zero)
    c = coefficients.get((0, 2), zero)
    d = coefficients.get((1, 0), zero)
    e = coefficients.get((0, 1), zero)
    constant = coefficients.get((0, 0), zero)
    total_degree = max(
        (sum(term.exponents) for term in polynomial.polynomial.terms),
        default=0,
    )
    if total_degree != 2:
        raise ValueError("rational conic polynomial must have total degree exactly two")
    px, py = (value.as_fraction() for value in point.values)
    if a * px * px + b * px * py + c * py * py + d * px + e * py + constant != 0:
        raise ValueError("supplied rational point must lie on the conic")

    gradient = (2 * a * px + b * py + d, b * px + 2 * c * py + e)
    # Closed-form determinant of the twice-projective symmetric matrix
    # ((2a, b, d), (b, 2c, e), (d, e, 2*constant)); nonzero exactly for a
    # smooth irreducible projective conic.
    determinant = (
        8 * a * c * constant
        + 2 * b * d * e
        - 2 * c * d * d
        - 2 * b * b * constant
        - 2 * a * e * e
    )
    if determinant == 0:
        raise ValueError("projective closure must be a smooth irreducible conic")
    if gradient == (zero, zero):
        raise ValueError("supplied point must be smooth on the affine conic")


def validate_rational_conic_request(
    polynomial: RationalPolynomial,
    point: VariablePoint,
    parameter: PolynomialVariable,
) -> None:
    """Admit one smooth conic-with-point before constructing its chart."""

    if len(polynomial.variables) != 2:
        raise ValueError(
            "rational conic parametrization requires exactly two variables"
        )
    if point.variables != polynomial.variables:
        raise ValueError("conic point must use the polynomial's complete ordered axis")
    if parameter in polynomial.variables:
        raise ValueError("parameter must be distinct from both conic variables")
    _require_source_bounds(polynomial, point)
    _require_smooth_conic_and_point(polynomial, point)
    _require_work_and_result_bounds(polynomial, point)


def _derive_rational_conic_parametrization(
    polynomial: RationalPolynomial,
    point: VariablePoint,
    parameter: PolynomialVariable,
) -> ConicParametrizationData:
    source, point_values, gradient, (a, b, c) = _source_data(polynomial, point)
    px, py = point_values
    fx, fy = gradient
    t = sympy.Symbol(parameter)
    direction_x = fx - fy * t
    direction_y = fy + fx * t
    gradient_square = fx**2 + fy**2
    denominator_expression = sympy.expand(
        a * direction_x**2 + b * direction_x * direction_y + c * direction_y**2
    )
    denominator = sympy.Poly(denominator_expression, t, domain=sympy.QQ)
    if denominator.degree() != 2:
        raise ValueError("smooth conic pencil must have a quadratic denominator")

    projective_x = sympy.Poly(
        sympy.expand(px * denominator_expression - gradient_square * direction_x),
        t,
        domain=sympy.QQ,
    )
    projective_y = sympy.Poly(
        sympy.expand(py * denominator_expression - gradient_square * direction_y),
        t,
        domain=sympy.QQ,
    )
    coordinate_x = rational_function_from_sympy(
        projective_x.as_expr() / denominator_expression,
        (parameter,),
        maximum_terms=MAX_CONIC_RESULT_TERMS,
    )
    coordinate_y = rational_function_from_sympy(
        projective_y.as_expr() / denominator_expression,
        (parameter,),
        maximum_terms=MAX_CONIC_RESULT_TERMS,
    )

    x, y = source.gens
    inverse_expression = (-fy * (x - px) + fx * (y - py)) / (
        fx * (x - px) + fy * (y - py)
    )
    inverse = rational_function_from_sympy(
        inverse_expression,
        polynomial.variables,
        maximum_terms=MAX_CONIC_RESULT_TERMS,
    )
    denominator_value = rational_polynomial_from_sympy(
        denominator.monic(),
        (parameter,),
        maximum_terms=MAX_CONIC_RESULT_TERMS,
    )
    require_polynomial_budget(
        denominator_value,
        maximum_terms=MAX_CONIC_RESULT_TERMS,
        maximum_exponent=2,
        maximum_coefficient_digits=MAX_CONIC_RESULT_DIGITS,
        label="finite-parameter denominator",
    )
    return ConicParametrizationData(
        coordinates=(coordinate_x, coordinate_y),
        inverse_parameter=inverse,
        finite_parameter_denominator=denominator_value,
    )


def derive_rational_conic_parametrization(
    polynomial: RationalPolynomial,
    point: VariablePoint,
    parameter: PolynomialVariable,
) -> ConicParametrizationData:
    """Construct the second intersection of a normalized line pencil."""

    return _derive_rational_conic_parametrization(polynomial, point, parameter)


__all__ = [
    "MAX_CONIC_INPUT_DIGITS",
    "MAX_CONIC_INTERMEDIATE_DIGITS",
    "MAX_CONIC_RESULT_DIGITS",
    "MAX_CONIC_RESULT_TERMS",
    "MAX_CONIC_TERMS",
    "ConicParametrizationData",
    "derive_rational_conic_parametrization",
    "validate_rational_conic_request",
]
