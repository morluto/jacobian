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


def _curve_discriminant(a: Fraction, b: Fraction) -> Fraction:
    """Compute Δ = -16(4A^3 + 27B^2)."""
    return -16 * (4 * a**3 + 27 * b**2)


def compute_discriminant(request: EllipticCurveRequest) -> CurveDiscriminantResult:
    """Compute the discriminant of a short Weierstrass curve."""
    a = request.curve.coefficient_a.as_fraction()
    b = request.curve.coefficient_b.as_fraction()
    disc = _curve_discriminant(a, b)
    return CurveDiscriminantResult(
        request=request,
        discriminant=CanonicalRational.from_fraction(disc),
        is_nonsingular=disc != 0,
    )


def check_point_on_curve(request: CurvePointRequest) -> PointOnCurveResult:
    """Check whether a point lies on a short Weierstrass curve."""
    a = request.curve.coefficient_a.as_fraction()
    b = request.curve.coefficient_b.as_fraction()
    x = request.point.x.as_fraction()
    y = request.point.y.as_fraction()
    lhs = y * y
    rhs = x * x * x + a * x + b
    return PointOnCurveResult(request=request, on_curve=lhs == rhs)


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
    a = request.curve.coefficient_a.as_fraction()
    b = request.curve.coefficient_b.as_fraction()
    first_point = request.first.point
    second_point = request.second.point
    x1 = first_point.x.as_fraction() if first_point else None
    y1 = first_point.y.as_fraction() if first_point else None
    x2 = second_point.x.as_fraction() if second_point else None
    y2 = second_point.y.as_fraction() if second_point else None

    # Unwrap parent-bearing operands; an identity contributes nothing.
    if x1 is None or y1 is None:
        if x2 is None or y2 is None:
            return EllipticCurvePointResult(curve=request.curve, at_infinity=True)
        return EllipticCurvePointResult(
            curve=request.curve,
            point=RationalAffinePoint(
                x=CanonicalRational.from_fraction(x2),
                y=CanonicalRational.from_fraction(y2),
            ),
        )
    if x2 is None or y2 is None:
        return EllipticCurvePointResult(
            curve=request.curve,
            point=RationalAffinePoint(
                x=CanonicalRational.from_fraction(x1),
                y=CanonicalRational.from_fraction(y1),
            ),
        )
    p1 = (x1, y1)
    p2 = (x2, y2)

    result = _point_add(a, b, p1, p2)
    if result is None:
        return EllipticCurvePointResult(curve=request.curve, at_infinity=True)
    x3, y3 = result
    return EllipticCurvePointResult(
        curve=request.curve,
        point=RationalAffinePoint(
            x=CanonicalRational.from_fraction(x3),
            y=CanonicalRational.from_fraction(y3),
        ),
    )


def scalar_multiply(
    request: ScalarMultiplicationRequest,
) -> ScalarMultiplicationResult:
    """Compute n*P on a short Weierstrass elliptic curve using double-and-add."""
    operand = request.point.point
    if request.scalar == 0 or request.point.at_infinity or operand is None:
        return ScalarMultiplicationResult(curve=request.curve, at_infinity=True)

    a = request.curve.coefficient_a.as_fraction()
    b = request.curve.coefficient_b.as_fraction()
    px = operand.x.as_fraction()
    py = operand.y.as_fraction()

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
                # The accumulated sum may cancel to infinity while higher
                # bits remain; keep scanning instead of discarding them.
                result = None if added is None else added
        if addend is not None:
            addend = _point_add(a, b, addend, addend)
        n >>= 1

    if result is None:
        return ScalarMultiplicationResult(curve=request.curve, at_infinity=True)
    return ScalarMultiplicationResult(
        curve=request.curve,
        point=RationalAffinePoint(
            x=CanonicalRational.from_fraction(result[0]),
            y=CanonicalRational.from_fraction(result[1]),
        ),
    )


__all__ = [
    "add_points",
    "check_point_on_curve",
    "compute_discriminant",
    "scalar_multiply",
]
