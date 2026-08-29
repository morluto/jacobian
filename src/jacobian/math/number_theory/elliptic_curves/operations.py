"""Domain-owned elliptic curve operations over QQ."""

from __future__ import annotations

from fractions import Fraction

from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math._rational_height import RationalHeight
from jacobian.math.number_theory.elliptic_curves._models import (
    MAX_SCALAR,
    CurveDiscriminantResult,
    EllipticCurvePointResult,
    PointOnCurveResult,
    RationalAffinePoint,
    ScalarMultiplicationResult,
    ShortWeierstrassCurve,
    _chord_step_heights,
    _doubling_lambda_height,
    _doubling_lambda_height_from_heights,
    _generic_lambda_height,
    _generic_lambda_height_from_heights,
    _point_heights,
    _require_group_law,
)


def _admission_error(exc: PydanticCustomError, location: tuple[str, ...]) -> None:
    raise OperationDomainValidationError(
        location=location, code=exc.type, message=exc.message()
    ) from exc


def _admit_point_addition(
    curve: ShortWeierstrassCurve,
    first: EllipticCurvePointResult,
    second: EllipticCurvePointResult,
) -> None:
    """Admit the group law and exact result-height envelope once per call."""
    if first.curve != curve or second.curve != curve:
        raise OperationDomainValidationError(
            location=("first", "second"),
            code="elliptic_curve.parent_curve_mismatch",
            message="operands must carry the supplied curve as their parent",
        )
    first_point = first.point
    second_point = second.point
    if first_point is None or second_point is None:
        try:
            _require_group_law(curve, ())
        except PydanticCustomError as exc:
            _admission_error(exc, ("curve",))
        return
    try:
        _require_group_law(curve, ())
        _require_group_law(curve, (first_point,))
    except PydanticCustomError as exc:
        _admission_error(
            exc,
            ("first",) if exc.type == "elliptic_curve.point_off_curve" else ("curve",),
        )
    try:
        _require_group_law(curve, (second_point,))
    except PydanticCustomError as exc:
        _admission_error(
            exc,
            ("second",) if exc.type == "elliptic_curve.point_off_curve" else ("curve",),
        )
    if first_point == second_point:
        if first_point.y.as_fraction() == 0:
            result = None
        else:
            result = _chord_step_heights(
                _doubling_lambda_height(curve, first_point),
                _point_heights(first_point),
                _point_heights(first_point),
            )
    elif first_point.x == second_point.x:
        result = None
    else:
        result = _chord_step_heights(
            _generic_lambda_height(first_point, second_point),
            _point_heights(first_point),
            _point_heights(second_point),
        )
    if result is not None and any(
        height.exceeds(MAX_CANONICAL_RATIONAL_DIGITS) for height in result
    ):
        raise OperationDomainValidationError(
            location=("first", "second"),
            code="elliptic_curve.point_addition_result_bound",
            message=(
                "point addition would produce coordinates exceeding the "
                "canonical result bound"
            ),
        )


def _admit_scalar_multiplication(
    curve: ShortWeierstrassCurve,
    point: EllipticCurvePointResult,
    scalar: int,
) -> None:
    """Admit the group law and double-and-add height envelope once per call."""
    if point.curve != curve:
        raise OperationDomainValidationError(
            location=("point",),
            code="elliptic_curve.parent_curve_mismatch",
            message="the operand must carry the supplied curve as its parent",
        )
    operand = point.point
    if point.at_infinity or operand is None:
        try:
            _require_group_law(curve, ())
        except PydanticCustomError as exc:
            _admission_error(exc, ("curve",))
        return
    try:
        _require_group_law(curve, (operand,))
    except PydanticCustomError as exc:
        location = (
            ("point",) if exc.type == "elliptic_curve.point_off_curve" else ("curve",)
        )
        _admission_error(exc, location)

    result: tuple[RationalHeight, RationalHeight] | None = None
    addend: tuple[RationalHeight, RationalHeight] | None = _point_heights(operand)
    addend_is_operand = True
    operand_y_is_zero = operand.y.as_fraction() == 0
    n = scalar
    while n > 0:
        if n & 1 and addend is not None:
            if result is None:
                result = addend
            else:
                lam = _generic_lambda_height_from_heights(result, addend)
                result = _chord_step_heights(lam, result, addend)
        if addend is not None:
            if addend_is_operand and operand_y_is_zero:
                addend = None
            else:
                lam = _doubling_lambda_height_from_heights(curve, addend)
                addend = _chord_step_heights(lam, addend, addend)
            addend_is_operand = False
        for slot in (result, addend):
            if slot is not None and any(
                height.exceeds(MAX_CANONICAL_RATIONAL_DIGITS) for height in slot
            ):
                raise OperationDomainValidationError(
                    location=("scalar",),
                    code="elliptic_curve.scalar_multiplication_result_bound",
                    message=(
                        "scalar multiplication would exceed the canonical result "
                        "height; reduce the scalar or use smaller coordinates"
                    ),
                )
        n >>= 1


def discriminant(curve: ShortWeierstrassCurve) -> CurveDiscriminantResult:
    """Compute the discriminant of a short Weierstrass curve."""
    disc = curve.discriminant()
    if (
        max(
            len(format_canonical_integer(abs(disc.numerator))),
            len(format_canonical_integer(disc.denominator)),
        )
        > MAX_CANONICAL_RATIONAL_DIGITS
    ):
        raise OperationDomainValidationError(
            location=("curve",),
            code="elliptic_curve.discriminant_result_bound",
            message=(
                "curve coefficients would produce a discriminant exceeding the "
                "canonical result bound"
            ),
        )
    return CurveDiscriminantResult._from_kernel(
        curve=curve,
        discriminant=CanonicalRational.from_fraction(disc),
        is_nonsingular=disc != 0,
    )


def point_on_curve(
    curve: ShortWeierstrassCurve, point: RationalAffinePoint
) -> PointOnCurveResult:
    """Check whether a point lies on a short Weierstrass curve."""
    a = curve.coefficient_a.as_fraction()
    b = curve.coefficient_b.as_fraction()
    x = point.x.as_fraction()
    y = point.y.as_fraction()
    lhs = y * y
    rhs = x * x * x + a * x + b
    return PointOnCurveResult._from_kernel(
        curve=curve,
        point=point,
        on_curve=lhs == rhs,
    )


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
    curve: ShortWeierstrassCurve,
    first: EllipticCurvePointResult,
    second: EllipticCurvePointResult,
) -> EllipticCurvePointResult:
    """Add two points on a short Weierstrass elliptic curve."""
    _admit_point_addition(curve, first, second)
    a = curve.coefficient_a.as_fraction()
    b = curve.coefficient_b.as_fraction()
    first_point = first.point
    second_point = second.point
    x1 = first_point.x.as_fraction() if first_point else None
    y1 = first_point.y.as_fraction() if first_point else None
    x2 = second_point.x.as_fraction() if second_point else None
    y2 = second_point.y.as_fraction() if second_point else None

    # Unwrap parent-bearing operands; an identity contributes nothing.
    if x1 is None or y1 is None:
        if x2 is None or y2 is None:
            return EllipticCurvePointResult._from_kernel(curve, None)
        return EllipticCurvePointResult._from_kernel(
            curve,
            RationalAffinePoint(
                x=CanonicalRational.from_fraction(x2),
                y=CanonicalRational.from_fraction(y2),
            ),
        )
    if x2 is None or y2 is None:
        return EllipticCurvePointResult._from_kernel(
            curve,
            RationalAffinePoint(
                x=CanonicalRational.from_fraction(x1),
                y=CanonicalRational.from_fraction(y1),
            ),
        )
    p1 = (x1, y1)
    p2 = (x2, y2)

    result = _point_add(a, b, p1, p2)
    if result is None:
        return EllipticCurvePointResult._from_kernel(curve, None)
    x3, y3 = result
    return EllipticCurvePointResult._from_kernel(
        curve,
        RationalAffinePoint(
            x=CanonicalRational.from_fraction(x3),
            y=CanonicalRational.from_fraction(y3),
        ),
    )


def scalar_multiply(
    curve: ShortWeierstrassCurve,
    point: EllipticCurvePointResult,
    scalar: int,
) -> ScalarMultiplicationResult:
    """Compute n*P on a short Weierstrass elliptic curve using double-and-add."""
    if type(scalar) is not int or not 0 <= scalar <= MAX_SCALAR:
        raise OperationDomainValidationError(
            location=("scalar",),
            code="elliptic_curve.scalar_out_of_range",
            message=f"scalar must be an integer between 0 and {MAX_SCALAR}",
        )
    _admit_scalar_multiplication(curve, point, scalar)
    operand = point.point
    if scalar == 0 or point.at_infinity or operand is None:
        return ScalarMultiplicationResult(curve=curve, at_infinity=True)

    a = curve.coefficient_a.as_fraction()
    b = curve.coefficient_b.as_fraction()
    px = operand.x.as_fraction()
    py = operand.y.as_fraction()

    result: tuple[Fraction, Fraction] | None = None
    # An infinite addend contributes nothing; doubling to the point at
    # infinity must not discard the accumulated result.
    addend: tuple[Fraction, Fraction] | None = (px, py)
    n = scalar

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
        return ScalarMultiplicationResult(curve=curve, at_infinity=True)
    return ScalarMultiplicationResult(
        curve=curve,
        point=RationalAffinePoint(
            x=CanonicalRational.from_fraction(result[0]),
            y=CanonicalRational.from_fraction(result[1]),
        ),
    )


__all__ = [
    "add_points",
    "discriminant",
    "point_on_curve",
    "scalar_multiply",
]
