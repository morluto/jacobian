"""Exact kernels for bounded integer-polynomial profiles and Mahler measure.

Quadratic arithmetic is kept private to this adapter. Results use the
repository's canonical rational and real-algebraic-root carriers, so a
consumer never has to learn a Mahler-specific scalar representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt, lcm
from time import monotonic
from typing import Literal

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS, CanonicalRational
from jacobian._execution import (
    current_request_execution,
    request_execution,
)
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS,
    RealAlgebraicValue,
)
from jacobian.math.polynomials._mahler_models import (
    MAX_MAHLER_COEFFICIENT_DIGITS,
    MAX_MAHLER_DEGREE,
    MahlerAlgebraicValue,
    MahlerMeasureResult,
    RealQuadraticRootProfileResult,
    ReciprocalProfileResult,
    RootLocation,
)
from jacobian.math.polynomials._models import IntegerPolynomial
from jacobian.math.polynomials.values import MAX_POLYNOMIAL_TERMS

__all__ = [
    "mahler_measure",
    "quadratic_root_profile",
    "reciprocal_profile",
]


@dataclass(frozen=True, slots=True)
class _QuadraticParts:
    """Private exact ``a + b*sqrt(d)`` arithmetic.

    The radicand is the content-normalized discriminant. It need not be
    square-free: ``_parts_to_value`` emits a primitive ``RealAlgebraicValue``.
    """

    rational: Fraction
    radical: Fraction
    radicand: int

    def nonnegative(self) -> bool:
        if self.radical == 0 or self.radicand == 0:
            return self.rational >= 0
        if self.rational >= 0 and self.radical >= 0:
            return True
        if self.rational < 0 and self.radical < 0:
            return False
        rational_square = self.rational * self.rational
        radical_square = self.radical * self.radical * self.radicand
        if self.rational >= 0:
            return rational_square >= radical_square
        return radical_square >= rational_square

    def multiply(self, other: _QuadraticParts) -> _QuadraticParts:
        if self.radicand == 0 or self.radical == 0:
            return _quadratic_parts_from_squarefree(
                self.rational * other.rational,
                self.rational * other.radical,
                1,
                other.radicand,
            )
        if other.radicand == 0 or other.radical == 0:
            return _quadratic_parts_from_squarefree(
                self.rational * other.rational,
                other.rational * self.radical,
                1,
                self.radicand,
            )
        if self.radicand != other.radicand:
            raise OperationDomainValidationError(
                location=("value",),
                code="polynomial.mahler_mixed_quadratic_fields",
                message="quadratic products require one shared discriminant field",
            )
        return _quadratic_parts_from_squarefree(
            self.rational * other.rational
            + self.radical * other.radical * self.radicand,
            self.rational * other.radical + other.rational * self.radical,
            1,
            self.radicand,
        )


def _quadratic_parts_from_squarefree(
    rational: Fraction,
    radical: Fraction,
    square_factor: int,
    squarefree: int,
) -> _QuadraticParts:
    if squarefree == 0 or radical == 0:
        return _QuadraticParts(rational, Fraction(0), 0)
    if squarefree == 1:
        return _QuadraticParts(rational + radical * square_factor, Fraction(0), 0)
    return _QuadraticParts(rational, radical * square_factor, squarefree)


def _parts_difference(left: _QuadraticParts, right: _QuadraticParts) -> _QuadraticParts:
    if left.radicand == 0 or left.radical == 0:
        return _quadratic_parts_from_squarefree(
            left.rational - right.rational,
            -right.radical,
            1,
            right.radicand,
        )
    if right.radicand == 0 or right.radical == 0:
        return _quadratic_parts_from_squarefree(
            left.rational - right.rational,
            left.radical,
            1,
            left.radicand,
        )
    if left.radicand != right.radicand:
        raise OperationDomainValidationError(
            location=("roots",),
            code="polynomial.mahler_mixed_quadratic_fields",
            message="quadratic ordering requires one shared discriminant field",
        )
    return _QuadraticParts(
        left.rational - right.rational,
        left.radical - right.radical,
        left.radicand,
    )


def _parts_less(left: _QuadraticParts, right: _QuadraticParts) -> bool:
    difference = _parts_difference(right, left)
    return difference.nonnegative() and not (
        difference.rational == 0 and difference.radical == 0
    )


def _parts_to_value(value: _QuadraticParts) -> MahlerAlgebraicValue:
    if value.radical == 0 or value.radicand == 0:
        return CanonicalRational.from_fraction(value.rational)
    denominator = lcm(
        value.rational.denominator,
        (2 * value.rational).denominator,
        (
            value.rational * value.rational
            - value.radical * value.radical * value.radicand
        ).denominator,
    )
    coefficients = (
        denominator,
        -2 * value.rational * denominator,
        (
            value.rational * value.rational
            - value.radical * value.radical * value.radicand
        )
        * denominator,
    )
    integer_coefficients = tuple(int(coefficient) for coefficient in coefficients)
    content = 0
    for coefficient in integer_coefficients:
        content = gcd(content, abs(coefficient))
    integer_coefficients = tuple(
        coefficient // content for coefficient in integer_coefficients
    )
    if integer_coefficients[0] < 0:
        integer_coefficients = tuple(
            -coefficient for coefficient in integer_coefficients
        )
    coefficient_digits = [
        len(format_canonical_integer(abs(coefficient)))
        for coefficient in integer_coefficients
    ]
    if max(coefficient_digits) > MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="polynomial.mahler_algebraic_result_bound",
            message=(
                "quadratic algebraic result exceeds its admitted coefficient "
                "digit bound"
            ),
        )
    return RealAlgebraicValue._from_admitted_polynomial(
        polynomial=integer_coefficients,
        real_root_index=1 if value.radical > 0 else 0,
    )


def _require_nonzero_polynomial(
    coefficients: tuple[int, ...], *, location: tuple[str | int, ...]
) -> None:
    if not any(coefficients):
        raise OperationDomainValidationError(
            location=location,
            code="polynomial.mahler_zero_polynomial",
            message="the zero polynomial has no content or root profile",
        )


# Content and reciprocal profiles copy source coefficients into multiple result
# fields. Charge those retained copies before construction so a carrier-valid
# high-height polynomial cannot explode after the kernel finishes. 66-term
# inexpensive cases remain well below this envelope.
MAX_PROFILE_RESULT_DIGITS = 8_000_000
_CANONICAL_INTEGER_LIMIT = 10**MAX_CANONICAL_INTEGER_DIGITS


def _retained_integer_digits(value: int) -> int:
    if value == 0:
        return 1
    return (abs(value).bit_length() * 30103) // 100000 + 1


def _coefficient_digit_total(coefficients: tuple[int, ...]) -> int:
    return sum(_retained_integer_digits(coefficient) for coefficient in coefficients)


def _admit_profile_result_digits(
    units: int, *, location: tuple[str | int, ...], code: str
) -> None:
    if units > MAX_PROFILE_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=location,
            code=code,
            message="the retained profile coefficients exceed the exact output bound",
        )


def _admit_mahler_coefficient_digits(polynomial: IntegerPolynomial) -> None:
    """Admit Mahler/quadratic coefficient height once after canonical parsing."""

    if any(
        len(format_canonical_integer(abs(coefficient))) > MAX_MAHLER_COEFFICIENT_DIGITS
        for coefficient in polynomial.coefficients
    ):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.mahler_coefficient_bound",
            message="a profile polynomial coefficient exceeds the admitted digit bound",
        )


def _exceeds_canonical_integer(value: int) -> bool:
    magnitude = abs(value)
    if magnitude.bit_length() <= MAX_CANONICAL_INTEGER_DIGITS:
        return False
    return bool(magnitude >= _CANONICAL_INTEGER_LIMIT)


def _require_integer_polynomial(polynomial: object) -> IntegerPolynomial:
    if not isinstance(polynomial, IntegerPolynomial):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.mahler_polynomial_type",
            message="Mahler-family operations require a canonical integer polynomial",
        )
    coefficients = polynomial.coefficients
    if not isinstance(coefficients, tuple) or not coefficients:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.mahler_polynomial_shape",
            message="a canonical integer polynomial has at least one coefficient",
        )
    if any(type(coefficient) is not int for coefficient in coefficients):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.mahler_polynomial_coefficients",
            message="Mahler-family coefficients must be exact integers",
        )
    if len(coefficients) > 1 and coefficients[0] == 0:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.mahler_polynomial_shape",
            message="a canonical integer polynomial omits leading zeros",
        )
    if len(coefficients) > MAX_POLYNOMIAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.mahler_carrier_term_bound",
            message="the integer polynomial exceeds the shared coefficient-carrier envelope",
        )
    if any(_exceeds_canonical_integer(coefficient) for coefficient in coefficients):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.mahler_carrier_integer_digits",
            message="a coefficient exceeds the canonical integer representation envelope",
        )
    return polynomial


def reciprocal_profile(polynomial: IntegerPolynomial) -> ReciprocalProfileResult:
    polynomial = _require_integer_polynomial(polynomial)
    coefficients = polynomial.coefficients
    _require_nonzero_polynomial(coefficients, location=("polynomial",))
    source_digits = _coefficient_digit_total(coefficients)
    # reversed_coefficients copies every source coefficient; the pair ledger
    # retains each coefficient once more, and the middle term of odd length
    # appears twice in its pair.
    extra = max(_retained_integer_digits(coefficient) for coefficient in coefficients)
    leading_digits = _retained_integer_digits(coefficients[0])
    constant_digits = _retained_integer_digits(coefficients[-1])
    _admit_profile_result_digits(
        2 * source_digits + extra + leading_digits + constant_digits,
        location=("polynomial",),
        code="polynomial.reciprocal_profile_result_digits",
    )
    degree = len(coefficients) - 1
    leading, constant = coefficients[0], coefficients[-1]
    state: Literal["RECIPROCAL", "ANTIRECIPROCAL", "NEITHER"]
    if constant != 0 and all(
        coefficients[index] == coefficients[degree - index]
        for index in range(degree + 1)
    ):
        state = "RECIPROCAL"
    elif all(
        coefficients[index] == -coefficients[degree - index]
        for index in range(degree + 1)
    ):
        state = "ANTIRECIPROCAL"
    else:
        state = "NEITHER"
    pair_count = (degree + 2) // 2
    return ReciprocalProfileResult._from_kernel(
        degree=degree,
        reversed_coefficients=tuple(reversed(coefficients)),
        state=state,
        leading_coefficient=leading,
        constant_coefficient=constant,
        coefficient_pair_ledger=tuple(
            (coefficients[index], coefficients[degree - index])
            for index in range(pair_count)
        ),
    )


def _unit_disk_location_of_parts(value: _QuadraticParts) -> RootLocation:
    if value.radical == 0 or value.radicand == 0:
        magnitude = abs(value.rational)
        return (
            "ON_UNIT_CIRCLE"
            if magnitude == 1
            else "OUTSIDE_UNIT_DISK"
            if magnitude > 1
            else "INSIDE_UNIT_DISK"
        )
    squared = _quadratic_parts_from_squarefree(
        value.rational * value.rational
        + value.radical * value.radical * value.radicand,
        2 * value.rational * value.radical,
        1,
        value.radicand,
    )
    difference = _parts_difference(
        squared, _QuadraticParts(Fraction(1), Fraction(0), 0)
    )
    if difference.rational == 0 and difference.radical == 0:
        return "ON_UNIT_CIRCLE"
    return "OUTSIDE_UNIT_DISK" if difference.nonnegative() else "INSIDE_UNIT_DISK"


def _quadratic_root_data(
    coefficients: tuple[int, int, int],
) -> tuple[
    Literal["DISTINCT_REAL", "DOUBLE_REAL", "COMPLEX_CONJUGATE"],
    tuple[_QuadraticParts, ...],
    tuple[RootLocation, ...],
]:
    content = gcd(gcd(abs(coefficients[0]), abs(coefficients[1])), abs(coefficients[2]))
    a, b, c = (coefficient // content for coefficient in coefficients)
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        squared = Fraction(c, a)
        locations: tuple[RootLocation, ...]
        if squared < 0:
            locations = ("UNRESOLVED",)
        else:
            locations = (
                "ON_UNIT_CIRCLE"
                if squared == 1
                else "OUTSIDE_UNIT_DISK"
                if squared > 1
                else "INSIDE_UNIT_DISK",
            )
        return "COMPLEX_CONJUGATE", (), locations
    if discriminant == 0:
        root = _QuadraticParts(Fraction(-b, 2 * a), Fraction(0), 0)
        return "DOUBLE_REAL", (root,), (_unit_disk_location_of_parts(root),)
    square = isqrt(discriminant)
    if square * square == discriminant:
        candidates = (
            _QuadraticParts(Fraction(-b - square, 2 * a), Fraction(0), 0),
            _QuadraticParts(Fraction(-b + square, 2 * a), Fraction(0), 0),
        )
    else:
        candidates = (
            _QuadraticParts(Fraction(-b, 2 * a), Fraction(-1, 2 * a), discriminant),
            _QuadraticParts(Fraction(-b, 2 * a), Fraction(1, 2 * a), discriminant),
        )
    # The two roots are in one field.  Compare the exact algebraic values,
    # rather than relying on the sign of ``a`` in the quadratic formula.
    first, second = candidates
    roots = (second, first) if _parts_less(second, first) else (first, second)
    return (
        "DISTINCT_REAL",
        roots,
        tuple(_unit_disk_location_of_parts(root) for root in roots),
    )


def _admit_quadratic_profile(polynomial: IntegerPolynomial) -> None:
    _admit_mahler_coefficient_digits(polynomial)
    if len(polynomial.coefficients) > MAX_MAHLER_DEGREE + 1:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.mahler_degree_bound",
            message=f"a profile polynomial has degree at most {MAX_MAHLER_DEGREE}",
        )
    if len(polynomial.coefficients) != 3:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.mahler_quadratic_degree",
            message="a real quadratic profile needs exactly three coefficients",
        )


def _quadratic_root_profile(
    polynomial: IntegerPolynomial,
) -> RealQuadraticRootProfileResult:
    _admit_quadratic_profile(polynomial)
    a, b, c = polynomial.coefficients
    if a == 0:
        raise OperationDomainValidationError(
            location=("polynomial", 0),
            code="polynomial.mahler_quadratic_leading",
            message="a real quadratic profile needs a nonzero leading coefficient",
        )
    root_kind, private_roots, root_locations = _quadratic_root_data((a, b, c))
    return RealQuadraticRootProfileResult(
        polynomial=IntegerPolynomial(coefficients=(a, b, c)),
        discriminant=b * b - 4 * a * c,
        root_kind=root_kind,
        sum_of_roots=CanonicalRational.from_fraction(Fraction(-b, a)),
        product_of_roots=CanonicalRational.from_fraction(Fraction(c, a)),
        roots=tuple(_parts_to_value(root) for root in private_roots),
        complex_pair_squared_modulus=(
            CanonicalRational.from_fraction(Fraction(c, a))
            if root_kind == "COMPLEX_CONJUGATE"
            else None
        ),
        root_locations=root_locations,
    )


def _abs_parts(value: _QuadraticParts) -> _QuadraticParts:
    return (
        value
        if value.nonnegative()
        else _QuadraticParts(-value.rational, -value.radical, value.radicand)
    )


def _admit_mahler_measure(polynomial: IntegerPolynomial) -> None:
    _admit_mahler_coefficient_digits(polynomial)
    if len(polynomial.coefficients) > MAX_MAHLER_DEGREE + 1:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.mahler_degree_bound",
            message=f"a profile polynomial has degree at most {MAX_MAHLER_DEGREE}",
        )
    if len(polynomial.coefficients) > 3:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.mahler_degree_bound",
            message="the Mahler measure needs degree at most two",
        )
    if polynomial.coefficients[0] == 0:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.mahler_leading_nonzero",
            message="the Mahler measure needs a nonzero leading coefficient",
        )


def _mahler_measure(polynomial: IntegerPolynomial) -> MahlerMeasureResult:
    _admit_mahler_measure(polynomial)
    coefficients = polynomial.coefficients
    leading = coefficients[0]
    roots: tuple[_QuadraticParts, ...]
    root_kind: Literal["DISTINCT_REAL", "DOUBLE_REAL", "COMPLEX_CONJUGATE"]
    if len(coefficients) == 1:
        roots = ()
        root_kind = "DISTINCT_REAL"
        root_locations: tuple[RootLocation, ...] = ()
    elif len(coefficients) == 2:
        roots = (_QuadraticParts(Fraction(-coefficients[1], leading), Fraction(0), 0),)
        root_kind = "DISTINCT_REAL"
        root_locations = tuple(_unit_disk_location_of_parts(root) for root in roots)
    else:
        root_kind, roots, root_locations = _quadratic_root_data(
            (coefficients[0], coefficients[1], coefficients[2])
        )
    if any(location == "UNRESOLVED" for location in root_locations):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.mahler_unresolved_location",
            message="the root-location ledger is incomplete, so no Mahler-measure claim is made",
        )
    outside = _QuadraticParts(Fraction(1), Fraction(0), 0)
    if root_kind == "COMPLEX_CONJUGATE":
        if root_locations[0] == "OUTSIDE_UNIT_DISK":
            outside = _QuadraticParts(
                abs(Fraction(coefficients[2], coefficients[0])), Fraction(0), 0
            )
        root_ledger = root_locations * 2
    elif root_kind == "DOUBLE_REAL":
        if root_locations[0] == "OUTSIDE_UNIT_DISK":
            outside = _abs_parts(roots[0]).multiply(_abs_parts(roots[0]))
        root_ledger = root_locations * 2
    else:
        for root, location in zip(roots, root_locations, strict=True):
            if location == "OUTSIDE_UNIT_DISK":
                outside = outside.multiply(_abs_parts(root))
        root_ledger = root_locations
    measure = _QuadraticParts(Fraction(abs(leading)), Fraction(0), 0).multiply(outside)
    outside_value = _parts_to_value(outside)
    measure_value = _parts_to_value(measure)
    return MahlerMeasureResult(
        polynomial=IntegerPolynomial(coefficients=coefficients),
        degree=len(coefficients) - 1,
        leading_coefficient=leading,
        root_locations=root_ledger,
        outside_root_product=outside_value,
        mahler_measure=measure_value,
    )


def quadratic_root_profile(
    polynomial: IntegerPolynomial,
) -> RealQuadraticRootProfileResult:
    polynomial = _require_integer_polynomial(polynomial)
    if current_request_execution() is None:
        with request_execution(monotonic()):
            return _quadratic_root_profile(polynomial)
    return _quadratic_root_profile(polynomial)


def mahler_measure(polynomial: IntegerPolynomial) -> MahlerMeasureResult:
    polynomial = _require_integer_polynomial(polynomial)
    if current_request_execution() is None:
        with request_execution(monotonic()):
            return _mahler_measure(polynomial)
    return _mahler_measure(polynomial)
