"""Typed trigonometric-rational normalization over QQ(i) Laurent polynomials."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.number_theory.number_fields import GaussianRational
from jacobian.math.number_theory.number_fields.values import (
    MAX_GAUSSIAN_RATIONAL_COMPONENT_DIGITS,
)

MAX_TRIG_VARIABLES = 8
MAX_TRIG_AST_NODES = 128
MAX_TRIG_LAURENT_TERMS = 4_096
MAX_TRIG_EXPONENT = 4_096
MAX_TRIG_GCD_EXPONENT = 2 * MAX_TRIG_EXPONENT
_GAUSSIAN_COMPONENT_LIMIT = 10**MAX_GAUSSIAN_RATIONAL_COMPONENT_DIGITS


class IntegerAffineAngleForm(StrictModel):
    """``quarter_turns*pi/2 + sum(coefficients[j]*angles[j])``."""

    coefficients: tuple[int, ...] = Field(
        min_length=0,
        max_length=MAX_TRIG_VARIABLES,
    )
    quarter_turns: int = Field(default=0, ge=-4_096, le=4_096)


class TrigLiteral(StrictModel):
    kind: Literal["LITERAL"]
    value: CanonicalRational


class TrigSine(StrictModel):
    kind: Literal["SINE"]
    angle: IntegerAffineAngleForm


class TrigCosine(StrictModel):
    kind: Literal["COSINE"]
    angle: IntegerAffineAngleForm


class TrigAdd(StrictModel):
    kind: Literal["ADD"]
    children: tuple[TrigonometricRationalExpression, ...] = Field(
        min_length=1, max_length=32
    )


class TrigMultiply(StrictModel):
    kind: Literal["MULTIPLY"]
    children: tuple[TrigonometricRationalExpression, ...] = Field(
        min_length=1, max_length=32
    )


class TrigDivide(StrictModel):
    kind: Literal["DIVIDE"]
    numerator: TrigonometricRationalExpression
    denominator: TrigonometricRationalExpression


class TrigPower(StrictModel):
    kind: Literal["POWER"]
    base: TrigonometricRationalExpression
    exponent: int = Field(ge=0, le=32)


type TrigonometricRationalExpression = Annotated[
    TrigLiteral
    | TrigSine
    | TrigCosine
    | TrigAdd
    | TrigMultiply
    | TrigDivide
    | TrigPower,
    Field(discriminator="kind"),
]


class GaussianLaurentTerm(StrictModel):
    coefficient: GaussianRational
    exponents: tuple[int, ...] = Field(max_length=MAX_TRIG_VARIABLES)

    @model_validator(mode="after")
    def require_bounded_exponents(self) -> Self:
        if any(abs(value) > MAX_TRIG_EXPONENT for value in self.exponents):
            raise PydanticCustomError(
                "trigonometric.exponent_bound",
                "Laurent exponents exceed the admitted representation limit",
            )
        return self


class GaussianLaurentPolynomial(StrictModel):
    variables: tuple[str, ...] = Field(max_length=MAX_TRIG_VARIABLES)
    terms: tuple[GaussianLaurentTerm, ...] = Field(max_length=MAX_TRIG_LAURENT_TERMS)

    @model_validator(mode="after")
    def require_canonical_support(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise PydanticCustomError(
                "trigonometric.variable_axis", "variables must be unique"
            )
        supports = tuple(term.exponents for term in self.terms)
        if any(len(item) != len(self.variables) for item in supports):
            raise PydanticCustomError(
                "trigonometric.exponent_axis", "exponents must align with variables"
            )
        if supports != tuple(sorted(supports, reverse=True)) or len(
            set(supports)
        ) != len(supports):
            raise PydanticCustomError(
                "trigonometric.support_order",
                "support must be unique in descending order",
            )
        if any(term.coefficient == GaussianRational.zero() for term in self.terms):
            raise PydanticCustomError(
                "trigonometric.zero_term", "zero coefficients must be omitted"
            )
        return self


class TrigonometricRationalSource(StrictModel):
    variables: tuple[str, ...] = Field(max_length=MAX_TRIG_VARIABLES)
    expression: TrigonometricRationalExpression


class TrigonometricRationalNormalizeResult(StrictModel):
    numerator: GaussianLaurentPolynomial
    denominator: GaussianLaurentPolynomial
    denominator_nonzero: GaussianLaurentPolynomial


Gaussian = tuple[Fraction, Fraction]
Support = tuple[int, ...]
Polynomial = dict[Support, Gaussian]
RationalFunction = tuple[Polynomial, Polynomial]


def _admit_gaussian(value: Gaussian) -> Gaussian:
    if any(
        abs(component.numerator) >= _GAUSSIAN_COMPONENT_LIMIT
        or component.denominator >= _GAUSSIAN_COMPONENT_LIMIT
        for component in value
    ):
        raise OperationResourceAdmissionError(
            location=("expression",),
            code="trigonometric_rational.coefficient_bound",
            message="trigonometric Laurent coefficients exceed the admitted exact-output bound",
        )
    return value


def _gadd(left: Gaussian, right: Gaussian) -> Gaussian:
    return _admit_gaussian((left[0] + right[0], left[1] + right[1]))


def _gmul(left: Gaussian, right: Gaussian) -> Gaussian:
    return _admit_gaussian(
        (
            left[0] * right[0] - left[1] * right[1],
            left[0] * right[1] + left[1] * right[0],
        )
    )


def _gdiv(left: Gaussian, right: Gaussian) -> Gaussian:
    norm = right[0] * right[0] + right[1] * right[1]
    if not norm:
        raise ZeroDivisionError
    return _admit_gaussian(
        (
            (left[0] * right[0] + left[1] * right[1]) / norm,
            (left[1] * right[0] - left[0] * right[1]) / norm,
        )
    )


def _gzero(value: Gaussian) -> bool:
    return not value[0] and not value[1]


def _poly_add(left: Polynomial, right: Polynomial) -> Polynomial:
    result = dict(left)
    for support, coefficient in right.items():
        result[support] = _gadd(
            result.get(support, (Fraction(), Fraction())), coefficient
        )
        if _gzero(result[support]):
            del result[support]
    return result


def _poly_mul(left: Polynomial, right: Polynomial) -> Polynomial:
    if len(left) * len(right) > MAX_TRIG_LAURENT_TERMS:
        _refuse_growth()
    result: Polynomial = {}
    for a_support, a_coefficient in left.items():
        for b_support, b_coefficient in right.items():
            support = tuple(a + b for a, b in zip(a_support, b_support, strict=True))
            _admit_support(support)
            result[support] = _gadd(
                result.get(support, (Fraction(), Fraction())),
                _gmul(a_coefficient, b_coefficient),
            )
            if _gzero(result[support]):
                del result[support]
    if len(result) > MAX_TRIG_LAURENT_TERMS:
        _refuse_growth()
    return result


def _refuse_growth() -> None:
    raise OperationResourceAdmissionError(
        location=("expression",),
        code="trigonometric_rational.expansion_bound",
        message="trigonometric Laurent expansion exceeds the admitted support or exponent bound",
    )


def _admit_support(support: Support) -> Support:
    if any(abs(value) > MAX_TRIG_EXPONENT for value in support):
        _refuse_growth()
    return support


def _admit_gcd_support(support: Support) -> Support:
    if any(abs(value) > MAX_TRIG_GCD_EXPONENT for value in support):
        _refuse_growth()
    return support


def _one(axis: int) -> Polynomial:
    return {(0,) * axis: (Fraction(1), Fraction())}


def _scale(polynomial: Polynomial, scalar: Gaussian) -> Polynomial:
    return {
        support: value
        for support, coefficient in polynomial.items()
        if not _gzero(value := _gmul(coefficient, scalar))
    }


def _power(value: RationalFunction, exponent: int, axis: int) -> RationalFunction:
    result = (_one(axis), _one(axis))
    base = value
    while exponent:
        if exponent & 1:
            result = (_poly_mul(result[0], base[0]), _poly_mul(result[1], base[1]))
        exponent >>= 1
        if exponent:
            base = (_poly_mul(base[0], base[0]), _poly_mul(base[1], base[1]))
    return result


def _root_of_unity(quarter_turns: int) -> Gaussian:
    return (
        (Fraction(1), Fraction()),
        (Fraction(), Fraction(1)),
        (Fraction(-1), Fraction()),
        (Fraction(), Fraction(-1)),
    )[quarter_turns % 4]


def _trig(angle: IntegerAffineAngleForm, axis: int, *, sine: bool) -> RationalFunction:
    if len(angle.coefficients) != axis:
        raise PydanticCustomError(
            "trigonometric.angle_axis", "angle coefficients must align with variables"
        )
    forward = tuple(angle.coefficients)
    backward = tuple(-value for value in forward)
    _admit_support(forward)
    _admit_support(backward)
    phase = _root_of_unity(angle.quarter_turns)
    inverse_phase = (phase[0], -phase[1])
    if sine:
        numerator = _poly_add(
            {forward: phase}, {backward: (-inverse_phase[0], -inverse_phase[1])}
        )
        denominator = _scale(_one(axis), (Fraction(), Fraction(2)))
    else:
        numerator = _poly_add({forward: phase}, {backward: inverse_phase})
        denominator = _scale(_one(axis), (Fraction(2), Fraction()))
    return numerator, denominator


def _evaluate(
    expression: TrigonometricRationalExpression, axis: int, nodes: list[int]
) -> RationalFunction:
    nodes[0] += 1
    if nodes[0] > MAX_TRIG_AST_NODES:
        _refuse_growth()
    if isinstance(expression, TrigLiteral):
        return _scale(_one(axis), (expression.value.as_fraction(), Fraction())), _one(
            axis
        )
    if isinstance(expression, TrigSine):
        return _trig(expression.angle, axis, sine=True)
    if isinstance(expression, TrigCosine):
        return _trig(expression.angle, axis, sine=False)
    if isinstance(expression, TrigPower):
        return _power(
            _evaluate(expression.base, axis, nodes), expression.exponent, axis
        )
    if isinstance(expression, TrigDivide):
        left = _evaluate(expression.numerator, axis, nodes)
        right = _evaluate(expression.denominator, axis, nodes)
        if not right[0]:
            raise PydanticCustomError(
                "trigonometric.zero_denominator",
                "division by the identically zero expression is undefined",
            )
        return _poly_mul(left[0], right[1]), _poly_mul(left[1], right[0])
    values = [_evaluate(child, axis, nodes) for child in expression.children]
    result = (
        (_one(axis), _one(axis))
        if isinstance(expression, TrigMultiply)
        else ({}, _one(axis))
    )
    for numerator, denominator in values:
        if isinstance(expression, TrigMultiply):
            result = _poly_mul(result[0], numerator), _poly_mul(result[1], denominator)
        else:
            result = (
                _poly_add(
                    _poly_mul(result[0], denominator), _poly_mul(numerator, result[1])
                ),
                _poly_mul(result[1], denominator),
            )
    return result


def _canonicalize(numerator: Polynomial, denominator: Polynomial) -> RationalFunction:
    if not denominator:
        raise PydanticCustomError(
            "trigonometric.zero_denominator", "denominator is identically zero"
        )
    if not numerator:
        return {}, {(0,) * len(next(iter(denominator))): (Fraction(1), Fraction())}
    shift = tuple(
        min(support[index] for support in denominator)
        for index in range(len(next(iter(denominator))))
    )
    numerator = {
        _admit_support(
            tuple(value - shift[index] for index, value in enumerate(support))
        ): coefficient
        for support, coefficient in numerator.items()
    }
    denominator = {
        _admit_support(
            tuple(value - shift[index] for index, value in enumerate(support))
        ): coefficient
        for support, coefficient in denominator.items()
    }
    leading = denominator[max(denominator)]
    numerator = {
        support: _gdiv(coefficient, leading)
        for support, coefficient in numerator.items()
    }
    denominator = {
        support: _gdiv(coefficient, leading)
        for support, coefficient in denominator.items()
    }
    return numerator, denominator


def _sympy_coefficient(value: Gaussian) -> Any:
    """Convert one exact Gaussian rational to the pinned backend domain."""

    from sympy import I, Rational

    return Rational(value[0].numerator, value[0].denominator) + I * Rational(
        value[1].numerator, value[1].denominator
    )


def _gaussian_from_sympy(value: Any) -> Gaussian:
    """Convert a backend Gaussian-rational coefficient without string parsing."""

    real, imaginary = value.as_real_imag()
    if not real.is_Rational or not imaginary.is_Rational:
        raise TypeError("trigonometric backend returned a non-rational coefficient")
    return _admit_gaussian(
        (
            Fraction(int(real.p), int(real.q)),
            Fraction(int(imaginary.p), int(imaginary.q)),
        )
    )


def _polynomial_from_sympy(value: Any) -> Polynomial:
    return {
        tuple(int(exponent) for exponent in exponents): _gaussian_from_sympy(
            coefficient
        )
        for exponents, coefficient in value.terms()
    }


def _reduce_common_laurent_factor(
    numerator: Polynomial, denominator: Polynomial
) -> RationalFunction:
    """Cancel the exact common Laurent factor before canonical normalization."""

    if not numerator:
        return _canonicalize(numerator, denominator)
    axis = len(next(iter(denominator)))
    if axis == 0 or len(denominator) == 1:
        return _canonicalize(numerator, denominator)

    # Shift both Laurent polynomials into an ordinary polynomial ring.  This
    # is multiplication by one common torus monomial and does not change the
    # rational function or its nonzero locus.
    minimum = tuple(
        min(
            support[index]
            for polynomial in (numerator, denominator)
            for support in polynomial
        )
        for index in range(axis)
    )
    shifted_numerator = {
        _admit_gcd_support(
            tuple(value - minimum[index] for index, value in enumerate(support))
        ): coefficient
        for support, coefficient in numerator.items()
    }
    shifted_denominator = {
        _admit_gcd_support(
            tuple(value - minimum[index] for index, value in enumerate(support))
        ): coefficient
        for support, coefficient in denominator.items()
    }
    if len(shifted_numerator) * len(shifted_denominator) > MAX_TRIG_LAURENT_TERMS:
        _refuse_growth()

    from sympy import Poly, Symbol
    from sympy.polys.domains import QQ_I

    symbols = tuple(Symbol(f"x{index}") for index in range(axis))
    left = Poly.from_dict(
        {
            support: _sympy_coefficient(coefficient)
            for support, coefficient in shifted_numerator.items()
        },
        *symbols,
        domain=QQ_I,
    )
    right = Poly.from_dict(
        {
            support: _sympy_coefficient(coefficient)
            for support, coefficient in shifted_denominator.items()
        },
        *symbols,
        domain=QQ_I,
    )
    common = left.gcd(right)
    return _canonicalize(
        _polynomial_from_sympy(left.exquo(common)),
        _polynomial_from_sympy(right.exquo(common)),
    )


def _canonicalize_nonzero_locus(denominator: Polynomial) -> Polynomial:
    """Normalize a source denominator while keeping its bounded Laurent support."""

    shift = tuple(
        min(support[index] for support in denominator)
        for index in range(len(next(iter(denominator))))
    )
    shifted_supports = tuple(
        tuple(value - shift[index] for index, value in enumerate(support))
        for support in denominator
    )
    if all(
        abs(value) <= MAX_TRIG_EXPONENT
        for support in shifted_supports
        for value in support
    ):
        return _canonicalize(_one(len(shift)), denominator)[1]

    # A monomial is nonzero everywhere on the algebraic torus. If the
    # canonical min-shift would exceed the output envelope, retain the source
    # signed supports and normalize only the scalar unit.
    leading = denominator[max(denominator)]
    return {
        support: _gdiv(coefficient, leading)
        for support, coefficient in denominator.items()
    }


def _wire(
    variables: tuple[str, ...], polynomial: Polynomial
) -> GaussianLaurentPolynomial:
    return GaussianLaurentPolynomial(
        variables=variables,
        terms=tuple(
            GaussianLaurentTerm(
                coefficient=GaussianRational.from_fractions(*coefficient),
                exponents=support,
            )
            for support, coefficient in sorted(polynomial.items(), reverse=True)
        ),
    )


def normalize_trigonometric_rational(
    request: TrigonometricRationalSource,
) -> TrigonometricRationalNormalizeResult:
    if len(set(request.variables)) != len(request.variables):
        raise PydanticCustomError(
            "trigonometric.variable_axis", "variables must be unique"
        )
    axis = len(request.variables)
    raw_numerator, raw_denominator = _evaluate(request.expression, axis, [0])
    numerator, denominator = _reduce_common_laurent_factor(
        raw_numerator, raw_denominator
    )
    denominator_nonzero = _canonicalize_nonzero_locus(raw_denominator)
    denominator_wire = _wire(request.variables, denominator)
    return TrigonometricRationalNormalizeResult(
        numerator=_wire(request.variables, numerator),
        denominator=denominator_wire,
        denominator_nonzero=_wire(request.variables, denominator_nonzero),
    )
