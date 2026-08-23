"""Domain-owned elliptic curve operations over QQ."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.elliptic_curves._models import (
    CurveDiscriminantResult,
    CurvePointRequest,
    EllipticCurvePointAdditionRequest,
    EllipticCurvePointResult,
    EllipticCurveRequest,
    PointOnCurveResult,
    RationalAffinePoint,
    ScalarMultiplicationRequest,
    ScalarMultiplicationResult,
)


def _rational_from_frac(value: Fraction) -> CanonicalRational:
    """Convert a Python Fraction to a CanonicalRational."""
    return CanonicalRational.from_integer_ratio(value.numerator, value.denominator)


def _frac_from_rational(value) -> Fraction:
    """Convert a CanonicalRational to a Python Fraction."""
    return value.as_fraction()


def _curve_discriminant(a: Fraction, b: Fraction) -> Fraction:
    """Compute Δ = -16(4A^3 + 27B^2)."""
    return -16 * (4 * a**3 + 27 * b**2)


def compute_discriminant(request: EllipticCurveRequest) -> CurveDiscriminantResult:
    """Compute the discriminant of a short Weierstrass curve."""
    a = _frac_from_rational(request.curve.coefficient_a)
    b = _frac_from_rational(request.curve.coefficient_b)
    disc = _curve_discriminant(a, b)
    return CurveDiscriminantResult(
        discriminant=_rational_from_frac(disc),
        is_nonsingular=disc != 0,
    )


def check_point_on_curve(request: CurvePointRequest) -> PointOnCurveResult:
    """Check whether a point lies on a short Weierstrass curve."""
    a = _frac_from_rational(request.curve.coefficient_a)
    b = _frac_from_rational(request.curve.coefficient_b)
    x = _frac_from_rational(request.point.x)
    y = _frac_from_rational(request.point.y)
    lhs = y * y
    rhs = x * x * x + a * x + b
    return PointOnCurveResult(on_curve=lhs == rhs)


def _point_add(
    a: Fraction,
    b: Fraction,
    p1: tuple[Fraction, Fraction],
    p2: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction] | None:
    """Add two points on y^2 = x^3 + Ax + B.

    Returns None if the result is the point at infinity.
    Raises ValueError if a point is not on the curve.
    """
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2:
        if y1 == y2:
            if y1 == 0:
                return None  # 2P = O for P of order 2
            lam = (3 * x1 * x1 + a) / (2 * y1)
        else:
            return None  # P + (-P) = O
    else:
        lam = (y2 - y1) / (x2 - x1)

    x3 = lam * lam - x1 - x2
    y3 = lam * (x1 - x3) - y1
    return x3, y3


def add_points(
    request: EllipticCurvePointAdditionRequest,
) -> EllipticCurvePointResult:
    """Add two points on a short Weierstrass elliptic curve."""
    a = _frac_from_rational(request.curve.coefficient_a)
    b = _frac_from_rational(request.curve.coefficient_b)
    x1 = _frac_from_rational(request.first.x)
    y1 = _frac_from_rational(request.first.y)
    x2 = _frac_from_rational(request.second.x)
    y2 = _frac_from_rational(request.second.y)

    result = _point_add(a, b, (x1, y1), (x2, y2))
    if result is None:
        return EllipticCurvePointResult(at_infinity=True)
    x3, y3 = result
    return EllipticCurvePointResult(
        point=RationalAffinePoint(
            x=_rational_from_frac(x3),
            y=_rational_from_frac(y3),
        ),
    )


def scalar_multiply(
    request: ScalarMultiplicationRequest,
) -> ScalarMultiplicationResult:
    """Compute n*P on a short Weierstrass elliptic curve using double-and-add."""
    if request.scalar == 0:
        return ScalarMultiplicationResult(curve=request.curve, at_infinity=True)

    a = _frac_from_rational(request.curve.coefficient_a)
    b = _frac_from_rational(request.curve.coefficient_b)
    px = _frac_from_rational(request.point.x)
    py = _frac_from_rational(request.point.y)

    result: tuple[Fraction, Fraction] | None = None
    # An infinite addend contributes nothing; doubling to the point at
    # infinity must not discard the accumulated result.
    addend: tuple[Fraction, Fraction] | None = (px, py)
    n = request.scalar

    while n > 0:
        if n & 1 and addend is not None:
            if result is None:
                result = addend
            else:
                added = _point_add(a, b, result, addend)
                if added is None:
                    # The accumulated sum cancelled to infinity, but higher bits
                    # remain; continue scanning rather than discarding them.
                    result = None
                else:
                    result = added
        if addend is not None:
            addend = _point_add(a, b, addend, addend)
        n >>= 1

    if result is None:
        return ScalarMultiplicationResult(curve=request.curve, at_infinity=True)
    return ScalarMultiplicationResult(
        curve=request.curve,
        point=RationalAffinePoint(
            x=_rational_from_frac(result[0]),
            y=_rational_from_frac(result[1]),
        ),
    )


__all__ = [
    "add_points",
    "check_point_on_curve",
    "compute_discriminant",
    "scalar_multiply",
]
