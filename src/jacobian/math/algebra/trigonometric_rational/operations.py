"""Typed trigonometric-rational normalization over QQ(i) Laurent polynomials."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.algebra.trigonometric_rational._laurent_gcd_process import (
    cancel_common_factor,
)
from jacobian.math.number_theory.number_fields import GaussianRational
from jacobian.math.number_theory.number_fields.values import (
    MAX_GAUSSIAN_RATIONAL_COMPONENT_DIGITS,
)

MAX_TRIG_VARIABLES = 8
MAX_TRIG_AST_NODES = 128
MAX_TRIG_LAURENT_TERMS = 4_096
# Locus divisibility runs a synchronous exact division before the bounded GCD
# worker; bound the operand product so it cannot become an unbounded phase.
_MAX_LOCUS_DIVISIBILITY_TERMS = 65_536
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


@dataclass(frozen=True, slots=True)
class _Evaluated:
    """One evaluated subexpression with its structural factor atoms.

    ``loci`` collects the atomic factors of every division denominator in the
    subtree, so shared factors between two loci stay a single atom and the zero
    locus is represented by their least common multiple. ``numerator_atoms``
    and ``denominator_atoms`` are the atomic factors of this numerator and
    denominator, used to decompose a denominator when it becomes a locus.
    """

    numerator: Polynomial
    denominator: Polynomial
    loci: tuple[Polynomial, ...]
    numerator_atoms: tuple[Polynomial, ...]
    denominator_atoms: tuple[Polynomial, ...]


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
    return _admit_gaussian(_gmul_raw(left, right))


def _gmul_raw(left: Gaussian, right: Gaussian) -> Gaussian:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
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


def _gaussian_proportional(left: Polynomial, right: Polynomial) -> bool:
    if left.keys() != right.keys() or not left:
        return False
    if left == right:
        return True
    first = next(iter(left))
    scale_left, scale_right = left[first], right[first]
    if _gzero(scale_right):
        return False
    return all(
        _gmul_raw(left[support], scale_right) == _gmul_raw(right[support], scale_left)
        for support in left
    )


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


def _axis_stride(exponents: list[int]) -> int:
    origin = min(exponents)
    stride = 0
    for value in exponents:
        stride = gcd(stride, value - origin)
    return stride or 1


def _quotient_support_term_count(numerator: Polynomial, denominator: Polynomial) -> int:
    """Bound reduced numerator and denominator support, preserving lattice stride."""

    if not numerator or not denominator:
        return 0
    axis = len(next(iter(numerator)))
    numerator_total = 1
    denominator_total = 1
    for index in range(axis):
        n_exps = [support[index] for support in numerator]
        d_exps = [support[index] for support in denominator]
        stride = gcd(_axis_stride(n_exps), _axis_stride(d_exps))
        n_span = max(n_exps) - min(n_exps)
        d_span = max(d_exps) - min(d_exps)
        n_width = n_span - d_span
        if n_width < 0:
            n_width = n_span
        d_width = d_span - n_span
        if d_width < 0:
            d_width = d_span
        n_count = n_width // stride + 1
        d_count = d_width // stride + 1
        if (
            n_count > MAX_TRIG_LAURENT_TERMS
            or d_count > MAX_TRIG_LAURENT_TERMS
            or numerator_total > MAX_TRIG_LAURENT_TERMS // n_count
            or denominator_total > MAX_TRIG_LAURENT_TERMS // d_count
        ):
            return MAX_TRIG_LAURENT_TERMS + 1
        numerator_total *= n_count
        denominator_total *= d_count
    return max(numerator_total, denominator_total)


def _one(axis: int) -> Polynomial:
    return {(0,) * axis: (Fraction(1), Fraction())}


def _scale(polynomial: Polynomial, scalar: Gaussian) -> Polynomial:
    return {
        support: value
        for support, coefficient in polynomial.items()
        if not _gzero(value := _gmul(coefficient, scalar))
    }


def _power(
    value: _Evaluated,
    exponent: int,
    axis: int,
) -> _Evaluated:
    result_num, result_den = _one(axis), _one(axis)
    base_num, base_den = value.numerator, value.denominator
    remaining = exponent
    while remaining:
        if remaining & 1:
            result_num = _poly_mul(result_num, base_num)
            result_den = _poly_mul(result_den, base_den)
        remaining >>= 1
        if remaining:
            base_num = _poly_mul(base_num, base_num)
            base_den = _poly_mul(base_den, base_den)
    # ``X**n = 0`` if and only if ``X = 0``, so a positive exponent does not
    # change the zero locus and the numerator atoms are the base's atoms. A zero
    # exponent gives the constant one, whose numerator and denominator have no
    # factor atoms, though any restriction recorded inside the base remains.
    if exponent == 0:
        return _Evaluated(
            numerator=result_num,
            denominator=result_den,
            loci=value.loci,
            numerator_atoms=(),
            denominator_atoms=(),
        )
    return _Evaluated(
        numerator=result_num,
        denominator=result_den,
        loci=value.loci,
        numerator_atoms=value.numerator_atoms,
        denominator_atoms=value.denominator_atoms,
    )


def _root_of_unity(quarter_turns: int) -> Gaussian:
    return (
        (Fraction(1), Fraction()),
        (Fraction(), Fraction(1)),
        (Fraction(-1), Fraction()),
        (Fraction(), Fraction(-1)),
    )[quarter_turns % 4]


def _trig(angle: IntegerAffineAngleForm, axis: int, *, sine: bool) -> _Evaluated:
    if len(angle.coefficients) != axis:
        raise OperationDomainValidationError(
            location=("angle",),
            code="trigonometric.angle_axis",
            message="angle coefficients must align with variables",
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
    return _Evaluated(
        numerator=numerator,
        denominator=denominator,
        loci=(),
        numerator_atoms=(numerator,),
        denominator_atoms=(denominator,),
    )


def _evaluate(
    expression: TrigonometricRationalExpression, axis: int, nodes: list[int]
) -> _Evaluated:
    nodes[0] += 1
    if nodes[0] > MAX_TRIG_AST_NODES:
        _refuse_growth()
    if isinstance(expression, TrigLiteral):
        numerator = _scale(_one(axis), (expression.value.as_fraction(), Fraction()))
        denominator = _one(axis)
        return _Evaluated(
            numerator=numerator,
            denominator=denominator,
            loci=(),
            numerator_atoms=(),
            denominator_atoms=(),
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
        if not right.numerator:
            raise OperationDomainValidationError(
                location=("expression", "denominator"),
                code="trigonometric.zero_denominator",
                message="division by the identically zero expression is undefined",
            )
        # The denominator's zero locus is its numerator's zero locus (the
        # denominator's own denominator is a nonzero monomial): record those
        # atoms so a factor shared with another division stays a single atom.
        return _Evaluated(
            numerator=_poly_mul(left.numerator, right.denominator),
            denominator=_poly_mul(left.denominator, right.numerator),
            loci=(*left.loci, *right.loci, *right.numerator_atoms),
            numerator_atoms=(*left.numerator_atoms, *right.denominator_atoms),
            denominator_atoms=(*left.denominator_atoms, *right.numerator_atoms),
        )
    values = [_evaluate(child, axis, nodes) for child in expression.children]
    is_multiply = isinstance(expression, TrigMultiply)
    result_num, result_den = (
        (_one(axis), _one(axis)) if is_multiply else ({}, _one(axis))
    )
    atoms: tuple[Polynomial, ...] = ()
    loci: tuple[Polynomial, ...] = ()
    for value in values:
        loci = (*loci, *value.loci)
        if is_multiply:
            result_num = _poly_mul(result_num, value.numerator)
            result_den = _poly_mul(result_den, value.denominator)
            atoms = (*atoms, *value.numerator_atoms)
        else:
            result_num = _poly_add(
                _poly_mul(result_num, value.denominator),
                _poly_mul(value.numerator, result_den),
            )
            result_den = _poly_mul(result_den, value.denominator)
    if not is_multiply:
        atoms = (result_num,) if result_num else ()
    return _Evaluated(
        numerator=result_num,
        denominator=result_den,
        loci=loci,
        numerator_atoms=atoms,
        denominator_atoms=(result_den,),
    )


def _canonicalize(numerator: Polynomial, denominator: Polynomial) -> RationalFunction:
    if not denominator:
        raise OperationDomainValidationError(
            location=("denominator",),
            code="trigonometric.zero_denominator",
            message="denominator is identically zero",
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
    if (
        len(numerator) > MAX_TRIG_LAURENT_TERMS
        or len(denominator) > MAX_TRIG_LAURENT_TERMS
    ):
        _refuse_growth()
    return numerator, denominator


def _polynomial_payload(polynomial: Polynomial) -> dict[str, list[object]]:
    supports = list(polynomial)
    return {
        "supports": [list(support) for support in supports],
        "real_numerators": [
            str(polynomial[support][0].numerator) for support in supports
        ],
        "real_denominators": [
            str(polynomial[support][0].denominator) for support in supports
        ],
        "imag_numerators": [
            str(polynomial[support][1].numerator) for support in supports
        ],
        "imag_denominators": [
            str(polynomial[support][1].denominator) for support in supports
        ],
    }


def _polynomial_from_payload(payload: object) -> Polynomial:
    if not isinstance(payload, dict):
        raise RuntimeError(
            "trigonometric Laurent GCD worker returned a malformed polynomial"
        )
    supports = payload["supports"]
    result: Polynomial = {}
    for support, real_num, real_den, imag_num, imag_den in zip(
        supports,
        payload["real_numerators"],
        payload["real_denominators"],
        payload["imag_numerators"],
        payload["imag_denominators"],
        strict=True,
    ):
        result[tuple(int(value) for value in support)] = (
            Fraction(int(real_num), int(real_den)),
            Fraction(int(imag_num), int(imag_den)),
        )
    return result


def _reduce_common_laurent_factor(
    numerator: Polynomial, denominator: Polynomial
) -> RationalFunction:
    """Cancel the exact common Laurent factor before canonical normalization."""

    if not numerator:
        return _canonicalize(numerator, denominator)
    axis = len(next(iter(denominator)))
    if axis == 0 or len(numerator) == 1 or len(denominator) == 1:
        return _canonicalize(numerator, denominator)
    if _gaussian_proportional(numerator, denominator):
        first = next(iter(numerator))
        constant = _gdiv(numerator[first], denominator[first])
        return _canonicalize(_scale(_one(axis), constant), _one(axis))

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
    if shifted_numerator == shifted_denominator:
        unit = {(0,) * axis: (Fraction(1), Fraction())}
        return _canonicalize(unit, unit)
    if len(shifted_numerator) * len(shifted_denominator) > MAX_TRIG_LAURENT_TERMS:
        _refuse_growth()
    if _quotient_support_term_count(shifted_numerator, shifted_denominator) > (
        MAX_TRIG_LAURENT_TERMS
    ):
        _refuse_growth()

    response = cancel_common_factor(
        {
            "axis": axis,
            "left": _polynomial_payload(shifted_numerator),
            "right": _polynomial_payload(shifted_denominator),
        }
    )
    reduced_numerator = _polynomial_from_payload(response["left"])
    reduced_denominator = _polynomial_from_payload(response["right"])
    # The cancellation can expand the numerator beyond the preflight envelope
    # (for example a coupled denominator factor); _canonicalize re-admits the
    # reduced pair with the typed resource bound so an oversized result cannot
    # surface as an untyped model-validation failure.
    return _canonicalize(reduced_numerator, reduced_denominator)


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


def _scalar_unit(polynomial: Polynomial) -> Polynomial:
    """Divide out the leading coefficient, leaving the Laurent support unchanged."""

    leading = polynomial[max(polynomial)]
    if leading == (Fraction(1), Fraction()):
        return polynomial
    return {
        support: _gdiv(coefficient, leading)
        for support, coefficient in polynomial.items()
    }


def _divides(candidate: Polynomial, target: Polynomial) -> bool:
    """Whether one Laurent polynomial divides another over ``QQ(i)``.

    Used to drop locus factors already implied by another retained factor, so
    the zero-set union is represented by its square-free cover rather than a
    redundant product that can exceed the exponent envelope.
    """

    if candidate == target:
        return True
    # Bound the divisibility check: it runs synchronously before the GCD worker
    # starts, so decline a large pair rather than perform unbounded exact
    # division. Declining only skips the pruning optimization.
    if len(candidate) * len(target) > _MAX_LOCUS_DIVISIBILITY_TERMS:
        return False
    axis = len(next(iter(candidate)))
    # Test divisibility in the Laurent ring: divide the two exact expressions and
    # require the quotient to have no negative-exponent terms. Cancelling first
    # keeps the comparison exact and bounded by the operand sizes.
    from sympy import QQ_I, Poly, Symbol, cancel, fraction
    from sympy.polys.polyerrors import CoercionFailed

    symbol = Symbol("z0")
    shifted_candidate = _shift_to_zero(candidate, axis)
    shifted_target = _shift_to_zero(target, axis)
    if axis == 1 and max(shifted_candidate) > max(shifted_target):
        return False
    try:
        candidate_polynomial = _to_sympy_poly(shifted_candidate)
        target_polynomial = _to_sympy_poly(shifted_target)
    except (ValueError, TypeError, CoercionFailed):
        return False
    if candidate_polynomial.is_zero:
        return bool(target_polynomial.is_zero)
    if axis == 1:
        # In one variable, Laurent divisibility is ordinary polynomial
        # divisibility after shifting each operand's minimum exponent to zero.
        # A degree check rejects sparse high-frequency factors immediately;
        # a finite-field remainder cheaply rejects most nondivisors before
        # exact division, which can otherwise scan a long dense degree range.
        if _univariate_modular_remainder_nonzero(shifted_candidate, shifted_target):
            return False
        # exact polynomial division then avoids ``cancel(target/candidate)``,
        # which computes a full subresultant GCD even though only divisibility
        # is needed.
        if candidate_polynomial.degree() > target_polynomial.degree():
            return False
        _quotient, remainder = target_polynomial.div(candidate_polynomial)
        return bool(remainder.is_zero)

    candidate_expr = candidate_polynomial.as_expr()
    target_expr = target_polynomial.as_expr()
    _numerator, denominator = fraction(cancel(target_expr / candidate_expr))
    # The quotient is a Laurent polynomial exactly when the residual denominator
    # is a monomial ``z0**k``.
    try:
        return bool(Poly(denominator, symbol, domain=QQ_I).is_monomial)
    except CoercionFailed:
        return False


def _univariate_modular_remainder_nonzero(
    candidate: Polynomial, target: Polynomial
) -> bool:
    """Prove univariate nondivisibility from one good reduction modulo p.

    The map ``QQ(i) -> F_p`` sending ``i`` to a square root of ``-1`` is a
    ring homomorphism for primes ``p == 1 mod 4``. If an exact quotient existed,
    it would remain a quotient after every specialization where the candidate
    keeps its degree. A nonzero remainder therefore proves exact
    nondivisibility; zero only falls through to the exact check.
    """
    for prime in (5, 13, 17):
        imaginary = _sqrt_minus_one_mod_prime(prime)
        if imaginary is None:
            continue
        divisor = _reduce_univariate_polynomial_mod_prime(candidate, prime, imaginary)
        dividend = _reduce_univariate_polynomial_mod_prime(target, prime, imaginary)
        if divisor is None or dividend is None:
            continue
        remainder_nonzero = _modular_remainder_nonzero(divisor, dividend, prime)
        if remainder_nonzero is True:
            return True
    return False


def _sqrt_minus_one_mod_prime(prime: int) -> int | None:
    return next(
        (value for value in range(prime) if value * value % prime == prime - 1),
        None,
    )


def _reduce_univariate_polynomial_mod_prime(
    polynomial: Polynomial, prime: int, imaginary: int
) -> dict[int, int] | None:
    reduced: dict[int, int] = {}
    for (exponent,), coefficient in polynomial.items():
        components = []
        for component in coefficient:
            if component.denominator % prime == 0:
                return None
            components.append(
                component.numerator * pow(component.denominator, -1, prime) % prime
            )
        value = (components[0] + imaginary * components[1]) % prime
        if value:
            reduced[exponent] = value
    return reduced


def _modular_remainder_nonzero(
    divisor: dict[int, int], dividend: dict[int, int], prime: int
) -> bool | None:
    if not divisor or not dividend:
        return None
    leading_exponent = max(divisor)
    if leading_exponent > max(dividend):
        return True
    inverse_leading = pow(divisor[leading_exponent], -1, prime)
    remainder = dividend
    steps = 0
    while remainder and max(remainder) >= leading_exponent:
        exponent = max(remainder)
        quotient = remainder[exponent] * inverse_leading % prime
        shift = exponent - leading_exponent
        for divisor_exponent, divisor_coefficient in divisor.items():
            position = shift + divisor_exponent
            value = (
                remainder.get(position, 0) - quotient * divisor_coefficient
            ) % prime
            if value:
                remainder[position] = value
            else:
                remainder.pop(position, None)
            steps += 1
            if steps > _MAX_LOCUS_DIVISIBILITY_TERMS:
                return None
    return bool(remainder)


def _shift_to_zero(polynomial: Polynomial, axis: int) -> Polynomial:
    """Shift a Laurent polynomial so its minimum exponent is zero per axis."""

    minimum = tuple(
        min(support[index] for support in polynomial) for index in range(axis)
    )
    return {
        tuple(
            value - minimum[index] for index, value in enumerate(support)
        ): coefficient
        for support, coefficient in polynomial.items()
    }


def _to_sympy_poly(polynomial: Polynomial) -> Any:
    from sympy import I, Integer, Poly, Rational, Symbol

    axis = len(next(iter(polynomial)))
    symbols = tuple(Symbol(f"y{index}") for index in range(axis))
    expression = Integer(0)
    for support, coefficient in polynomial.items():
        term = Integer(1)
        for index, exponent in enumerate(support):
            term *= symbols[index] ** exponent
        expression += term * (
            Rational(coefficient[0].numerator, coefficient[0].denominator)
            + I * Rational(coefficient[1].numerator, coefficient[1].denominator)
        )
    return Poly(expression, *symbols, domain="QQ_I")


def _combine_loci(loci: tuple[Polynomial, ...], axis: int) -> Polynomial:
    """Combine locus atoms into the least common multiple of their zero sets.

    ``loci`` are the atomic factors of every division denominator, so a factor
    shared by two loci (for example ``P`` in ``P*sin(x)`` and ``P*cos(x)``)
    arrives as one atom and is charged once. Proportional and divisible atoms
    are pruned as before; the surviving atoms are multiplied.
    """

    unique: list[Polynomial] = []
    for polynomial in loci:
        if not polynomial:
            continue
        support = next(iter(polynomial))
        if len(polynomial) == 1 and not any(support):
            continue
        normalized = _scalar_unit(polynomial)
        if any(_gaussian_proportional(normalized, existing) for existing in unique):
            continue
        # The locus is the union of the factors' zero sets, and ``a | b`` means
        # ``b`` has the larger zero set (``b = 0`` whenever ``a = 0``). Keep the
        # factor with the larger zero set: skip ``normalized`` if a retained
        # factor is a multiple of it, and drop any retained factor that is a
        # multiple of ``normalized``.
        if any(_divides(normalized, existing) for existing in unique):
            continue
        unique = [existing for existing in unique if not _divides(existing, normalized)]
        unique.append(normalized)
    result = _one(axis)
    for polynomial in unique:
        result = _poly_mul(result, polynomial)
    return result


def normalize_trigonometric_rational(
    request: TrigonometricRationalSource,
) -> TrigonometricRationalNormalizeResult:
    if len(set(request.variables)) != len(request.variables):
        raise OperationDomainValidationError(
            location=("variables",),
            code="trigonometric.variable_axis",
            message="variables must be unique",
        )
    axis = len(request.variables)
    evaluated = _evaluate(request.expression, axis, [0])
    raw_numerator = evaluated.numerator
    raw_denominator = evaluated.denominator
    combined_loci = _combine_loci(evaluated.loci, axis)
    source_locus = combined_loci if combined_loci != _one(axis) else raw_denominator
    denominator_nonzero = _canonicalize_nonzero_locus(source_locus)
    numerator, denominator = _reduce_common_laurent_factor(
        raw_numerator, raw_denominator
    )
    denominator_wire = _wire(request.variables, denominator)
    return TrigonometricRationalNormalizeResult(
        numerator=_wire(request.variables, numerator),
        denominator=denominator_wire,
        denominator_nonzero=_wire(request.variables, denominator_nonzero),
    )
