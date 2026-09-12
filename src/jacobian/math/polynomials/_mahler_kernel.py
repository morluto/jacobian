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

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    current_request_execution,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory._certification_models import (
    CertifiedFactorizationRequest,
)
from jacobian.math.number_theory._factorization_kernels import factorize_certified
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS,
    RealAlgebraicValue,
)
from jacobian.math.polynomials._mahler_models import (
    MAX_MAHLER_RADICAND_BITS,
    MAX_MAHLER_RADICAND_DIGITS,
    ContentPrimitiveProfileRequest,
    ContentPrimitiveProfileResult,
    IntegerPolynomialProfileValue,
    IntegerPolynomialValue,
    MahlerAlgebraicValue,
    MahlerMeasureRequest,
    MahlerMeasureResult,
    RealQuadraticRootProfileRequest,
    RealQuadraticRootProfileResult,
    ReciprocalProfileRequest,
    ReciprocalProfileResult,
    RootLocation,
)

__all__ = [
    "content_primitive_profile",
    "mahler_measure",
    "quadratic_root_profile",
    "reciprocal_profile",
]

_SMALL_SQUAREFREE_FACTOR_LIMIT = 1_000_000
_MAX_FACTORIZATION_WORK = 1_000_000
_MAX_RESULT_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class _QuadraticParts:
    """Private exact ``a + b*sqrt(d)`` arithmetic with squarefree ``d``."""

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
            return _quadratic_parts(
                self.rational * other.rational,
                self.rational * other.radical,
                other.radicand,
            )
        if other.radicand == 0 or other.radical == 0:
            return _quadratic_parts(
                self.rational * other.rational,
                other.rational * self.radical,
                self.radicand,
            )
        if self.radicand != other.radicand:
            raise OperationDomainValidationError(
                location=("value",),
                code="polynomial.mahler_mixed_quadratic_fields",
                message="quadratic products require one shared squarefree field",
            )
        return _quadratic_parts(
            self.rational * other.rational
            + self.radical * other.radical * self.radicand,
            self.rational * other.radical + other.rational * self.radical,
            self.radicand,
        )


def _squarefree_parts(radicand: int) -> tuple[int, int]:
    """Return ``(square_factor, squarefree_radicand)`` under one budget."""

    if radicand < 1:
        raise ValueError("surd radicand must be positive")
    if radicand.bit_length() > MAX_MAHLER_RADICAND_BITS:
        raise OperationResourceAdmissionError(
            location=("radicand",),
            code="polynomial.mahler_surd_radicand_bound",
            message="quadratic discriminant exceeds the admitted factorization envelope",
        )
    if len(str(radicand)) > MAX_MAHLER_RADICAND_DIGITS:
        raise OperationResourceAdmissionError(
            location=("radicand",),
            code="polynomial.mahler_surd_radicand_digits",
            message="quadratic discriminant exceeds the admitted factorization digits",
        )
    estimated_work = radicand.bit_length() ** 3
    if estimated_work > _MAX_FACTORIZATION_WORK:
        raise OperationResourceAdmissionError(
            location=("radicand",),
            code="polynomial.mahler_factorization_work_bound",
            message="quadratic discriminant factorization exceeds the admitted work bound",
        )
    if radicand <= _SMALL_SQUAREFREE_FACTOR_LIMIT:
        square_factor = 1
        remaining = radicand
        factor = 2
        while factor * factor <= remaining:
            square = factor * factor
            while remaining % square == 0:
                remaining //= square
                square_factor *= factor
            factor += 1
        return square_factor, remaining
    try:
        decomposition = factorize_certified(
            CertifiedFactorizationRequest(value=radicand)
        )
    except (OperationExecutionCancelledError, OperationExecutionTimeoutError):
        raise
    except RuntimeError as exc:
        raise OperationResourceAdmissionError(
            location=("radicand",),
            code="polynomial.mahler_factorization_backend",
            message="the admitted factorization backend could not complete",
        ) from exc
    square_factor = 1
    squarefree_radicand = 1
    for factor_entry in decomposition.factors:
        square_factor *= factor_entry.prime ** (factor_entry.exponent // 2)
        squarefree_radicand *= factor_entry.prime ** (factor_entry.exponent % 2)
    return square_factor, squarefree_radicand


def _quadratic_parts(
    rational: Fraction, radical: Fraction, radicand: int
) -> _QuadraticParts:
    if radicand == 0 or radical == 0:
        return _QuadraticParts(rational, Fraction(0), 0)
    square_factor, squarefree = _squarefree_parts(radicand)
    if squarefree == 1:
        return _QuadraticParts(rational + radical * square_factor, Fraction(0), 0)
    return _QuadraticParts(rational, radical * square_factor, squarefree)


def _parts_difference(left: _QuadraticParts, right: _QuadraticParts) -> _QuadraticParts:
    if left.radicand == 0 or left.radical == 0:
        return _quadratic_parts(
            left.rational - right.rational,
            -right.radical,
            right.radicand,
        )
    if right.radicand == 0 or right.radical == 0:
        return _quadratic_parts(
            left.rational - right.rational,
            left.radical,
            left.radicand,
        )
    if left.radicand != right.radicand:
        raise OperationDomainValidationError(
            location=("roots",),
            code="polynomial.mahler_mixed_quadratic_fields",
            message="quadratic ordering requires one shared squarefree field",
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
        len(str(abs(coefficient))) for coefficient in integer_coefficients
    ]
    estimated_bytes = sum(digits + 8 for digits in coefficient_digits) + 128
    if (
        max(coefficient_digits) > MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS
        or estimated_bytes > _MAX_RESULT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("result",),
            code="polynomial.mahler_algebraic_result_bound",
            message="quadratic algebraic result exceeds its admitted height or byte bound",
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


def content_primitive_profile(
    request: ContentPrimitiveProfileRequest,
) -> ContentPrimitiveProfileResult:
    coefficients = request.polynomial.coefficients_descending
    _require_nonzero_polynomial(coefficients, location=("polynomial",))
    sign: Literal[-1, 1] = 1 if coefficients[0] > 0 else -1
    content = 0
    for coefficient in coefficients:
        content = gcd(content, abs(coefficient))
    primitive = tuple((sign * coefficient) // content for coefficient in coefficients)
    primitive_value = IntegerPolynomialProfileValue(coefficients_descending=primitive)
    reconstruction = tuple(sign * content * value for value in primitive)
    return ContentPrimitiveProfileResult(
        sign=sign,
        content=content,
        primitive_part=primitive_value,
        degree=primitive_value.degree,
        reconstruction=IntegerPolynomialValue(coefficients_descending=reconstruction),
    )


def reciprocal_profile(request: ReciprocalProfileRequest) -> ReciprocalProfileResult:
    coefficients = request.polynomial.coefficients_descending
    _require_nonzero_polynomial(coefficients, location=("polynomial",))
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
    return ReciprocalProfileResult(
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
    squared = _quadratic_parts(
        value.rational * value.rational
        + value.radical * value.radical * value.radicand,
        2 * value.rational * value.radical,
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
    a, b, c = coefficients
    discriminant = b * b - 4 * a * c
    if discriminant > 0 and discriminant.bit_length() > MAX_MAHLER_RADICAND_BITS:
        raise OperationResourceAdmissionError(
            location=("coefficients_descending",),
            code="polynomial.mahler_surd_radicand_bound",
            message="quadratic discriminant exceeds the admitted surd-factorization envelope",
        )
    if discriminant > 0 and len(str(discriminant)) > MAX_MAHLER_RADICAND_DIGITS:
        raise OperationResourceAdmissionError(
            location=("coefficients_descending",),
            code="polynomial.mahler_surd_radicand_digits",
            message="quadratic discriminant exceeds the admitted factorization digits",
        )
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
            _quadratic_parts(Fraction(-b, 2 * a), Fraction(-1, 2 * a), discriminant),
            _quadratic_parts(Fraction(-b, 2 * a), Fraction(1, 2 * a), discriminant),
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


def _quadratic_root_profile(
    request: RealQuadraticRootProfileRequest,
) -> RealQuadraticRootProfileResult:
    a, b, c = request.coefficients_descending
    if a == 0:
        raise OperationDomainValidationError(
            location=("coefficients_descending", 0),
            code="polynomial.mahler_quadratic_leading",
            message="a real quadratic profile needs a nonzero leading coefficient",
        )
    root_kind, private_roots, root_locations = _quadratic_root_data((a, b, c))
    return RealQuadraticRootProfileResult(
        coefficients_descending=(a, b, c),
        discriminant=b * b - 4 * a * c,
        root_kind=root_kind,
        sum_of_roots=(-b, a),
        product_of_roots=(c, a),
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


def _mahler_measure(request: MahlerMeasureRequest) -> MahlerMeasureResult:
    coefficients = request.coefficients_descending
    leading = coefficients[0]
    roots: tuple[_QuadraticParts, ...]
    if len(coefficients) == 2:
        roots = (_QuadraticParts(Fraction(-coefficients[1], leading), Fraction(0), 0),)
        root_kind: Literal["DISTINCT_REAL", "DOUBLE_REAL", "COMPLEX_CONJUGATE"] = (
            "DISTINCT_REAL"
        )
        root_locations = tuple(_unit_disk_location_of_parts(root) for root in roots)
        discriminant = 0
    else:
        root_kind, roots, root_locations = _quadratic_root_data(
            (coefficients[0], coefficients[1], coefficients[2])
        )
        discriminant = (
            coefficients[1] * coefficients[1] - 4 * coefficients[0] * coefficients[2]
        )
    if any(location == "UNRESOLVED" for location in root_locations):
        raise OperationResourceAdmissionError(
            location=("coefficients_descending",),
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
    if len(str(abs(leading))) + len(str(discriminant)) + 512 > _MAX_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="polynomial.mahler_result_bytes_bound",
            message="Mahler result exceeds its admitted output byte bound",
        )
    return MahlerMeasureResult(
        coefficients_descending=coefficients,
        degree=len(coefficients) - 1,
        leading_coefficient=leading,
        root_locations=root_ledger,
        outside_root_product=outside_value,
        mahler_measure=measure_value,
    )


def quadratic_root_profile(
    request: RealQuadraticRootProfileRequest,
) -> RealQuadraticRootProfileResult:
    if current_request_execution() is None:
        with request_execution(monotonic()):
            return _quadratic_root_profile(request)
    return _quadratic_root_profile(request)


def mahler_measure(request: MahlerMeasureRequest) -> MahlerMeasureResult:
    if current_request_execution() is None:
        with request_execution(monotonic()):
            return _mahler_measure(request)
    return _mahler_measure(request)
