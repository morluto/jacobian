"""Exact kernels for bounded integer-polynomial profiles and Mahler measure.

Everything here is exact integer, rational, or quadratic-surd arithmetic.  Root
locations compare an exact squared modulus against one, so no numerical
tolerance is ever authoritative, and an unseparated comparison is reported as
``UNRESOLVED`` rather than resolved by rounding.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd, isqrt
from time import monotonic
from typing import Literal

from jacobian._exact import CanonicalRational
from jacobian._execution import current_request_execution, request_execution
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory._certification_models import (
    CertifiedFactorizationRequest,
)
from jacobian.math.number_theory._factorization_kernels import factorize_certified
from jacobian.math.polynomials._mahler_models import (
    MAX_MAHLER_RADICAND_BITS,
    MAX_MAHLER_RADICAND_DIGITS,
    ContentPrimitiveProfileRequest,
    ContentPrimitiveProfileResult,
    IntegerPolynomialProfileValue,
    IntegerPolynomialValue,
    MahlerMeasureRequest,
    MahlerMeasureResult,
    QuadraticSurd,
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


def _squarefree_parts(radicand: int) -> tuple[int, int]:
    """Return ``(square_factor, squarefree_radicand)`` under one kernel budget."""

    if radicand < 1:
        raise ValueError("surd radicand must be positive")
    if radicand.bit_length() > MAX_MAHLER_RADICAND_BITS:
        raise OperationResourceAdmissionError(
            location=("radicand",),
            code="polynomial.mahler_surd_radicand_bound",
            message="quadratic-surd radicand exceeds the admitted factorization envelope",
        )
    if len(str(radicand)) > MAX_MAHLER_RADICAND_DIGITS:
        raise OperationResourceAdmissionError(
            location=("radicand",),
            code="polynomial.mahler_surd_radicand_digits",
            message="quadratic-surd radicand exceeds the admitted factorization digits",
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
    decomposition = factorize_certified(CertifiedFactorizationRequest(value=radicand))
    square_factor = 1
    squarefree_radicand = 1
    for factor_entry in decomposition.factors:
        square_factor *= factor_entry.prime ** (factor_entry.exponent // 2)
        squarefree_radicand *= factor_entry.prime ** (factor_entry.exponent % 2)
    return square_factor, squarefree_radicand


def quadratic_surd_from_fractions(
    rational: Fraction, radical: Fraction, radicand: int
) -> QuadraticSurd:
    """Canonicalize one quadratic surd through the bounded kernel adapter."""

    if radicand == 0 or radical == 0:
        return QuadraticSurd.rational(rational)
    square_factor, squarefree_radicand = _squarefree_parts(radicand)
    return QuadraticSurd.from_squarefree_parts(
        rational,
        radical,
        square_factor,
        squarefree_radicand,
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
    """Return sign, content, and the positive-leading primitive part."""

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
    """Return the exact reversal structure of one integer polynomial."""

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


def _unit_disk_location_of_surd(value: QuadraticSurd) -> RootLocation:
    """Exact classification of ``|a + b*sqrt(d)|`` against one."""

    a, b = value.as_fractions()
    if value.radicand == 0 or b == 0:
        magnitude = abs(a)
        return (
            "ON_UNIT_CIRCLE"
            if magnitude == 1
            else "OUTSIDE_UNIT_DISK"
            if magnitude > 1
            else "INSIDE_UNIT_DISK"
        )
    # |a + b*sqrt(d)|^2 = (a^2 + b^2 d) + 2 a b sqrt(d), which is nonnegative.
    squared = quadratic_surd_from_fractions(
        a * a + b * b * value.radicand, 2 * a * b, value.radicand
    )
    difference = quadratic_surd_from_fractions(
        squared.as_fractions()[0] - 1,
        squared.as_fractions()[1],
        value.radicand,
    )
    if difference.is_zero():
        return "ON_UNIT_CIRCLE"
    return "OUTSIDE_UNIT_DISK" if difference.is_nonnegative() else "INSIDE_UNIT_DISK"


def _quadratic_root_profile(
    request: RealQuadraticRootProfileRequest,
) -> RealQuadraticRootProfileResult:
    """Return exact roots and unit-disk locations of one real quadratic."""

    a, b, c = request.coefficients_descending
    if a == 0:
        raise OperationDomainValidationError(
            location=("coefficients_descending", 0),
            code="polynomial.mahler_quadratic_leading",
            message="a real quadratic profile needs a nonzero leading coefficient",
        )
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
    root_kind: Literal["DISTINCT_REAL", "DOUBLE_REAL", "COMPLEX_CONJUGATE"]
    roots: tuple[QuadraticSurd, ...]
    if discriminant > 0:
        root_kind = "DISTINCT_REAL"
        square = isqrt(discriminant)
        if square * square == discriminant:
            roots = (
                QuadraticSurd.rational(Fraction(-b - square, 2 * a)),
                QuadraticSurd.rational(Fraction(-b + square, 2 * a)),
            )
        else:
            roots = (
                quadratic_surd_from_fractions(
                    Fraction(-b, 2 * a), Fraction(-1, 2 * a), discriminant
                ),
                quadratic_surd_from_fractions(
                    Fraction(-b, 2 * a), Fraction(1, 2 * a), discriminant
                ),
            )
    elif discriminant == 0:
        root_kind = "DOUBLE_REAL"
        roots = (QuadraticSurd.rational(Fraction(-b, 2 * a)),)
    else:
        root_kind = "COMPLEX_CONJUGATE"
        roots = ()
    return RealQuadraticRootProfileResult(
        coefficients_descending=(a, b, c),
        discriminant=discriminant,
        root_kind=root_kind,
        sum_of_roots=(-b, a),
        product_of_roots=(c, a),
        roots=roots,
        complex_pair_squared_modulus=CanonicalRational.from_fraction(Fraction(c, a))
        if root_kind == "COMPLEX_CONJUGATE"
        else None,
        root_locations=(
            (_unit_disk_location_of_quadratic(None, root_kind, a, c),)
            if root_kind == "COMPLEX_CONJUGATE"
            else tuple(
                _unit_disk_location_of_quadratic(root, root_kind, a, c)
                for root in roots
            )
        ),
    )


def _unit_disk_location_of_quadratic(
    root: QuadraticSurd | None, root_kind: str, a: int, c: int
) -> RootLocation:
    if root_kind == "COMPLEX_CONJUGATE":
        # |root|^2 = c/a exactly; classify the square, then report the location.
        squared = Fraction(c, a)
        if squared < 0:
            return "UNRESOLVED"
        return (
            "ON_UNIT_CIRCLE"
            if squared == 1
            else "OUTSIDE_UNIT_DISK"
            if squared > 1
            else "INSIDE_UNIT_DISK"
        )
    assert root is not None
    return _unit_disk_location_of_surd(root)


def _abs_of_surd(value: QuadraticSurd) -> QuadraticSurd:
    """Exact ``|value|`` for one admitted real quadratic surd."""

    a, b = value.as_fractions()
    if value.radicand == 0 or b == 0:
        return QuadraticSurd.rational(abs(a))
    return (
        value
        if value.is_nonnegative()
        else QuadraticSurd.from_squarefree_parts(-a, -b, 1, value.radicand)
    )


def _mahler_measure(request: MahlerMeasureRequest) -> MahlerMeasureResult:
    """Return ``|a_d| * prod_i max(1, |alpha_i|)`` exactly for degree <= 2."""

    coefficients = request.coefficients_descending
    leading = coefficients[0]
    roots: tuple[QuadraticSurd, ...]
    root_locations: tuple[RootLocation, ...]
    roots_ledger: tuple[RootLocation, ...]
    if len(coefficients) == 2:
        # a x + c has the single rational root -c/a.
        root = QuadraticSurd.rational(Fraction(-coefficients[1], leading))
        root_locations = (_unit_disk_location_of_surd(root),)
        roots = (root,)
        product = Fraction(0)
        root_kind = "DISTINCT_REAL"
    else:
        profile = quadratic_root_profile(
            RealQuadraticRootProfileRequest(
                coefficients_descending=(
                    coefficients[0],
                    coefficients[1],
                    coefficients[2],
                )
            )
        )
        root_kind = profile.root_kind
        roots = profile.roots
        root_locations = profile.root_locations
        product = Fraction(*profile.product_of_roots)
    outside = QuadraticSurd.rational(Fraction(1))
    for location in root_locations:
        if location == "UNRESOLVED":
            raise OperationResourceAdmissionError(
                location=("coefficients_descending",),
                code="polynomial.mahler_unresolved_location",
                message=(
                    "the root-location ledger is incomplete, so no Mahler-measure "
                    "claim is made"
                ),
            )
    if root_kind == "COMPLEX_CONJUGATE":
        if root_locations[0] == "OUTSIDE_UNIT_DISK":
            # The two conjugate roots contribute |root|^2 = c/a in total.
            outside = QuadraticSurd.rational(abs(product))
        roots_ledger = root_locations * 2
    elif root_kind == "DOUBLE_REAL":
        if root_locations[0] == "OUTSIDE_UNIT_DISK":
            outside = _abs_of_surd(roots[0]).multiply(_abs_of_surd(roots[0]))
        roots_ledger = root_locations * 2
    else:
        for root, location in zip(roots, root_locations, strict=True):
            if location == "OUTSIDE_UNIT_DISK":
                outside = outside.multiply(_abs_of_surd(root))
        roots_ledger = root_locations
    measure = QuadraticSurd.rational(Fraction(abs(leading))).multiply(outside)
    return MahlerMeasureResult(
        coefficients_descending=coefficients,
        degree=len(coefficients) - 1,
        leading_coefficient=leading,
        root_locations=roots_ledger,
        outside_root_product=outside,
        mahler_measure=measure,
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
