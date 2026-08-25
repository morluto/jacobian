"""Typed wire contracts for elliptic curve operations over QQ."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math._rational_height import RationalHeight, sum_heights


class ShortWeierstrassCurve(StrictModel):
    """A short Weierstrass curve y^2 = x^3 + A*x + B over QQ."""

    coefficient_a: CanonicalRational
    coefficient_b: CanonicalRational

    def discriminant(self) -> Fraction:
        """Exact Δ = -16(4A³ + 27B²); zero marks a singular cubic."""
        a = self.coefficient_a.as_fraction()
        b = self.coefficient_b.as_fraction()
        return -16 * (4 * a**3 + 27 * b**2)


class EllipticCurveRequest(StrictModel):
    """Compute the discriminant of a short Weierstrass curve."""

    curve: ShortWeierstrassCurve

    @model_validator(mode="after")
    def require_discriminant_result_bound(self) -> Self:
        # A cancellation-blind height estimate over-rejects curves whose
        # large terms cancel exactly (A = -3t^2, B = 2t^3 gives
        # 4A^3 + 27B^2 = 0 exactly).  Compute the reduced exact
        # discriminant instead — three multiplications at the canonical
        # digit limits, bounded work — and admit on its actual magnitude.
        # Zero is admissible: the result reports singularity itself.
        discriminant = self.curve.discriminant()
        numerator_digits = len(format_canonical_integer(abs(discriminant.numerator)))
        denominator_digits = len(format_canonical_integer(discriminant.denominator))
        if (
            numerator_digits > MAX_CANONICAL_RATIONAL_DIGITS
            or denominator_digits > MAX_CANONICAL_RATIONAL_DIGITS
        ):
            raise PydanticCustomError(
                "elliptic_curve.discriminant_result_bound",
                "curve coefficients would produce a discriminant exceeding the canonical result bound",
            )
        return self


class CurveDiscriminantResult(StrictModel):
    """The discriminant Δ = -16(4A^3 + 27B^2) of its retained source curve."""

    request: EllipticCurveRequest
    discriminant: CanonicalRational
    is_nonsingular: bool

    @model_validator(mode="after")
    def require_consistent_nonsingularity(self) -> Self:
        if self.is_nonsingular is (self.discriminant.as_fraction() == 0):
            raise PydanticCustomError(
                "elliptic_curve.nonsingularity_mismatch",
                "nonsingularity must match a nonzero discriminant",
            )
        # The value must be derived from the retained curve: replay the
        # exact formula so a valid-looking payload cannot detach from any
        # computation.
        a = self.request.curve.coefficient_a.as_fraction()
        b = self.request.curve.coefficient_b.as_fraction()
        expected = -16 * (4 * a**3 + 27 * b**2)
        if self.discriminant.as_fraction() != expected:
            raise PydanticCustomError(
                "elliptic_curve.discriminant_source_mismatch",
                "discriminant must be the exact discriminant of the retained source curve",
            )
        return self


class RationalAffinePoint(StrictModel):
    """An affine rational point on an elliptic curve."""

    x: CanonicalRational
    y: CanonicalRational


class CurvePointRequest(StrictModel):
    """A curve and a point to check or operate on."""

    curve: ShortWeierstrassCurve
    point: RationalAffinePoint


class PointOnCurveResult(StrictModel):
    """Whether a point lies on its retained source curve."""

    request: CurvePointRequest
    on_curve: bool

    @model_validator(mode="after")
    def require_derived_predicate(self) -> Self:
        # Replay y² = x³ + Ax + B from the retained sources so positive or
        # negative conclusions are checkable and cannot be forged.
        x = self.request.point.x.as_fraction()
        y = self.request.point.y.as_fraction()
        a = self.request.curve.coefficient_a.as_fraction()
        b = self.request.curve.coefficient_b.as_fraction()
        if self.on_curve is not (y * y == x**3 + a * x + b):
            raise PydanticCustomError(
                "elliptic_curve.point_membership_mismatch",
                "on_curve must match the exact curve equation of the retained source",
            )
        return self


def _require_group_law(
    curve: ShortWeierstrassCurve,
    points: tuple[RationalAffinePoint, ...],
) -> None:
    """Enforce the advertised group-law domain at the typed boundary.

    The chord-and-tangent formulas compute on the curve only when the cubic
    is nonsingular and every operand satisfies y² = x³ + Ax + B.
    """
    if curve.discriminant() == 0:
        raise PydanticCustomError(
            "elliptic_curve.singular_curve",
            "curve must be nonsingular (nonzero discriminant)",
        )
    for point in points:
        x = point.x.as_fraction()
        y = point.y.as_fraction()
        if y * y != x**3 + curve.coefficient_a.as_fraction() * x + (
            curve.coefficient_b.as_fraction()
        ):
            raise PydanticCustomError(
                "elliptic_curve.point_off_curve",
                "point must lie on the curve",
            )


def _generic_lambda_height_from_heights(
    first: tuple[RationalHeight, RationalHeight],
    second: tuple[RationalHeight, RationalHeight],
) -> RationalHeight:
    """Height bound of lambda = (y2 - y1) / (x2 - x1) with symbolic operands."""
    dy = sum_heights((second[1], first[1]))
    dx = sum_heights((second[0], first[0]))
    return dy.quotient(dx)


def _doubling_lambda_height_from_heights(
    curve: ShortWeierstrassCurve, point: tuple[RationalHeight, RationalHeight]
) -> RationalHeight:
    """Height bound of lambda = (3x^2 + A) / (2y) for symbolic coordinates."""
    x, y = point
    three_x_squared = RationalHeight(
        2 * x.numerator_digits + 1, 2 * x.denominator_digits
    )
    numerator = sum_heights(
        (three_x_squared, RationalHeight.from_canonical(curve.coefficient_a))
    )
    return numerator.quotient(
        RationalHeight(y.numerator_digits + 1, y.denominator_digits)
    )


def _generic_lambda_height(
    first: RationalAffinePoint, second: RationalAffinePoint
) -> RationalHeight:
    """Height bound of lambda = (y2 - y1) / (x2 - x1)."""
    dy = sum_heights(
        (
            RationalHeight.from_canonical(second.y),
            RationalHeight.from_canonical(first.y),
        )
    )
    dx = sum_heights(
        (
            RationalHeight.from_canonical(second.x),
            RationalHeight.from_canonical(first.x),
        )
    )
    return dy.quotient(dx)


def _doubling_lambda_height(
    curve: ShortWeierstrassCurve, point: RationalAffinePoint
) -> RationalHeight:
    """Height bound of lambda = (3x^2 + A) / (2y)."""
    x = RationalHeight.from_canonical(point.x)
    three_x_squared = RationalHeight(
        2 * x.numerator_digits + 1, 2 * x.denominator_digits
    )
    numerator = sum_heights(
        (three_x_squared, RationalHeight.from_canonical(curve.coefficient_a))
    )
    y = RationalHeight.from_canonical(point.y)
    return numerator.quotient(
        RationalHeight(y.numerator_digits + 1, y.denominator_digits)
    )


def _chord_step_heights(
    lam: RationalHeight,
    first: tuple[RationalHeight, RationalHeight],
    second: tuple[RationalHeight, RationalHeight],
) -> tuple[RationalHeight, RationalHeight]:
    """Conservative coordinate heights of one chord-and-tangent output.

    With lambda bounded by ``lam``, x3 = lambda^2 - x1 - x2 and
    y3 = lambda * (x1 - x3) - y1 propagate through rational-height sums,
    products, and quotients.
    """
    lam_squared = lam.product(lam)
    x3 = sum_heights((lam_squared, first[0], second[0]))
    inner = sum_heights((first[0], x3))
    y3 = sum_heights((lam.product(inner), first[1]))
    return (
        RationalHeight(
            max(x3.numerator_digits, y3.numerator_digits),
            max(x3.denominator_digits, y3.denominator_digits),
        ),
        y3,
    )


def _point_heights(point: RationalAffinePoint) -> tuple[RationalHeight, RationalHeight]:
    return (
        RationalHeight.from_canonical(point.x),
        RationalHeight.from_canonical(point.y),
    )


class EllipticCurvePointResult(StrictModel):
    """The result of an elliptic curve point operation on its parent curve.

    The parent curve defines the group the result lives in: without it,
    identical coordinate pairs on different curves serialize to the same
    value and callers cannot feed the point back into another group-law
    operation.
    """

    curve: ShortWeierstrassCurve
    point: RationalAffinePoint | None = None
    at_infinity: bool = False

    @model_validator(mode="after")
    def require_consistent_point(self) -> Self:
        # One canonical infinity discriminator: at_infinity. Accepting a
        # second independent flag would let one mathematical value
        # serialize several ways and let downstream at_infinity readers
        # misread a validated infinity as finite-with-no-point.
        if self.point is not None and self.at_infinity:
            raise PydanticCustomError(
                "elliptic_curve.point_infinity_conflict",
                "a finite point and infinity are mutually exclusive",
            )
        if self.point is None and not self.at_infinity:
            raise PydanticCustomError(
                "elliptic_curve.point_missing",
                "must carry a finite point or indicate infinity",
            )
        if self.point is not None:
            x = self.point.x.as_fraction()
            y = self.point.y.as_fraction()
            a = self.curve.coefficient_a.as_fraction()
            b = self.curve.coefficient_b.as_fraction()
            if y * y != x**3 + a * x + b:
                raise PydanticCustomError(
                    "elliptic_curve.result_point_off_curve",
                    "result point must lie on the retained curve",
                )
        return self


class EllipticCurvePointAdditionRequest(StrictModel):
    """Add two points on a short Weierstrass elliptic curve.

    Both operands are parent-bearing curve-point values — exactly the shape
    the group-law producers return — so a doubling example's infinity result
    or any finite point result composes into this request unchanged.
    """

    curve: ShortWeierstrassCurve
    first: EllipticCurvePointResult
    second: EllipticCurvePointResult

    @model_validator(mode="after")
    def require_operands_in_the_same_group(self) -> Self:
        if self.first.curve != self.curve or self.second.curve != self.curve:
            raise PydanticCustomError(
                "elliptic_curve.parent_curve_mismatch",
                "operands must carry this request's curve as their parent",
            )
        return self

    @model_validator(mode="after")
    def require_group_law(self) -> Self:
        first_point = self.first.point
        second_point = self.second.point
        # An identity operand contributes nothing and adds no height, but
        # the group law itself still requires a nonsingular curve.
        if first_point is None or second_point is None:
            _require_group_law(self.curve, ())
            return self
        _require_group_law(self.curve, (first_point, second_point))
        if first_point == second_point:
            if first_point.y.as_fraction() == 0:
                # A point of order two doubles to the identity: no tangent
                # slope exists and no coordinate height can grow.
                result = None
            else:
                result = _chord_step_heights(
                    _doubling_lambda_height(self.curve, first_point),
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
            raise PydanticCustomError(
                "elliptic_curve.point_addition_result_bound",
                "point addition would produce coordinates exceeding the canonical result bound",
            )
        return self


class ScalarMultiplicationRequest(StrictModel):
    """Compute n*P on a short Weierstrass elliptic curve."""

    curve: ShortWeierstrassCurve
    point: EllipticCurvePointResult
    scalar: int = Field(ge=0, le=10_000)

    @model_validator(mode="after")
    def require_operand_in_the_same_group(self) -> Self:
        if self.point.curve != self.curve:
            raise PydanticCustomError(
                "elliptic_curve.parent_curve_mismatch",
                "the operand must carry this request's curve as its parent",
            )
        return self

    @model_validator(mode="after")
    def require_group_law(self) -> Self:
        operand = self.point.point
        # An identity operand contributes nothing, but the group law itself
        # still requires a nonsingular curve.
        if self.point.at_infinity or operand is None:
            _require_group_law(self.curve, ())
            return self
        _require_group_law(self.curve, (operand,))
        # Propagate coordinate heights through the same double-and-add scan
        # the kernel performs: each bit doubles the addend and adds it to the
        # accumulator on a set bit, and every step's chord-and-tangent output
        # is bounded by rational-height propagation.  The naive n^2 digit
        # heuristic both over-rejects and admits doublings whose exact
        # coordinates exceed the canonical limit, so derive the budget from
        # the recurrence.
        #
        # Each slot carries the finite/infinity state the group law actually
        # produces.  An order-two point's first doubling lands on the identity
        # (its y vanishes), the identity absorbs every later doubling, and no
        # slope height is propagated through an infinity slot.  Intermediates
        # whose state is not decidable from the request stay conservatively
        # finite, which can only overestimate heights.
        result: tuple[RationalHeight, RationalHeight] | None = None
        addend: tuple[RationalHeight, RationalHeight] | None = _point_heights(operand)
        addend_is_operand = True
        operand_y_is_zero = operand.y.as_fraction() == 0
        n = self.scalar
        while n > 0:
            if n & 1 and addend is not None:
                if result is None:
                    result = addend
                else:
                    lam = _generic_lambda_height_from_heights(result, addend)
                    result = _chord_step_heights(lam, result, addend)
            if addend is not None:
                if addend_is_operand and operand_y_is_zero:
                    # 2P = O for a point of order two; the identity absorbs
                    # every later doubling of this addend.
                    addend = None
                else:
                    lam = _doubling_lambda_height_from_heights(self.curve, addend)
                    addend = _chord_step_heights(lam, addend, addend)
                addend_is_operand = False
            for slot in (result, addend):
                if slot is not None and any(
                    height.exceeds(MAX_CANONICAL_RATIONAL_DIGITS) for height in slot
                ):
                    raise PydanticCustomError(
                        "elliptic_curve.scalar_multiplication_result_bound",
                        "scalar multiplication would exceed the canonical result height; reduce the scalar or use smaller coordinates",
                    )
            n >>= 1
        return self


class ScalarMultiplicationResult(EllipticCurvePointResult):
    """The result of scalar multiplication n*P on its retained parent curve.

    The shared parent-bearing curve-point value: a scalar-multiplication
    result passes unchanged as the operand of any later group-law request.
    """


# Membership replay is inherited from EllipticCurvePointResult's validator.
ScalarMultiplicationResult.model_rebuild()


__all__ = [
    "CurveDiscriminantResult",
    "CurvePointRequest",
    "EllipticCurvePointAdditionRequest",
    "EllipticCurvePointResult",
    "EllipticCurveRequest",
    "PointOnCurveResult",
    "RationalAffinePoint",
    "ScalarMultiplicationRequest",
    "ScalarMultiplicationResult",
    "ShortWeierstrassCurve",
]
