"""Canonical bounded real algebraic values and exact order."""

from __future__ import annotations

from math import gcd
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer

# A degree-eight value covers every singular value of a 2 by 2 matrix over a
# real quadratic field.  The coefficient budget also bounds exact comparison:
# the product of two defining polynomials has degree at most sixteen and
# coefficient height below 2,002 decimal digits.  Mignotte's root-separation
# bound then needs fewer than 32,768 decimal digits for rational isolating
# endpoints, so every accepted comparison remains representable by the shared
# canonical scalar envelope.
MAX_REAL_ALGEBRAIC_DEGREE = 8
MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS = 1_000

RealAlgebraicOrder = Literal["LT", "EQ", "GT"]


def _sympy_polynomial(value: RealAlgebraicValue):  # type: ignore[no-untyped-def]
    import sympy

    x = sympy.Symbol("x")
    return sympy.Poly.from_list(
        [parse_canonical_integer(coefficient) for coefficient in value.polynomial],
        gens=x,
        domain=sympy.ZZ,
    )


def _strict_root_count(poly, lower, upper) -> int:  # type: ignore[no-untyped-def]
    """Count roots in an open interval, or at one singleton endpoint."""

    if lower == upper:
        return int(poly.eval(lower) == 0)
    count = int(poly.count_roots(lower, upper))
    if poly.eval(lower) == 0:
        count -= 1
    if poly.eval(upper) == 0:
        count -= 1
    return count


def _rational(value) -> CanonicalRational:  # type: ignore[no-untyped-def]
    import sympy

    rational = sympy.Rational(value)
    return CanonicalRational(
        num=format_canonical_integer(int(rational.p)),
        den=format_canonical_integer(int(rational.q)),
    )


class RealAlgebraicValue(StrictModel):
    """One real algebraic number in canonical minimal-polynomial form.

    ``polynomial`` is the primitive irreducible polynomial in ``ZZ[x]`` with
    positive leading coefficient, listed in descending degree.  The
    zero-based ``real_root_index`` selects one of its real roots in increasing
    order.  This pair uniquely determines the value and its real embedding.
    """

    real_algebraic_schema_version: Literal["1"] = "1"
    polynomial: tuple[CanonicalInteger, ...] = Field(
        min_length=2,
        max_length=MAX_REAL_ALGEBRAIC_DEGREE + 1,
        description=(
            "Primitive irreducible ZZ[x] coefficients in descending degree, "
            "with positive leading coefficient and at most 1,000 digits each."
        ),
        examples=[["1", "0", "-2"]],
    )
    real_root_index: StrictInt = Field(
        ge=0,
        description=(
            "Zero-based index of the selected real root when all real roots of "
            "the minimal polynomial are ordered increasingly."
        ),
        examples=[1],
    )

    @model_validator(mode="after")
    def require_canonical_real_root(self) -> Self:
        if any(
            len(coefficient.lstrip("-")) > MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS
            for coefficient in self.polynomial
        ):
            raise ValueError(
                "real algebraic polynomial coefficients exceed the "
                f"{MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS}-digit bound"
            )
        coefficients = tuple(
            parse_canonical_integer(coefficient) for coefficient in self.polynomial
        )
        if coefficients[0] <= 0:
            raise ValueError(
                "real algebraic minimal polynomial must have positive leading coefficient"
            )
        content = 0
        for coefficient in coefficients:
            content = gcd(content, abs(coefficient))
        if content != 1:
            raise ValueError(
                "real algebraic minimal polynomial must be primitive over ZZ"
            )

        polynomial = _sympy_polynomial(self)
        if polynomial.is_irreducible is not True:
            raise ValueError(
                "real algebraic minimal polynomial must be irreducible over QQ"
            )
        real_root_count = len(polynomial.intervals())
        if self.real_root_index >= real_root_count:
            raise ValueError(
                "real_root_index must select an existing real root of the minimal polynomial"
            )
        return self


class RationalIsolatingInterval(StrictModel):
    """A canonical rational interval isolating one real polynomial root."""

    lower: CanonicalRational
    upper: CanonicalRational
    interval_type: Literal["OPEN", "SINGLETON"]

    @model_validator(mode="after")
    def require_interval_convention(self) -> Self:
        lower = self.lower.as_fraction()
        upper = self.upper.as_fraction()
        if lower > upper:
            raise ValueError("isolating interval lower endpoint must not exceed upper")
        expected = "SINGLETON" if lower == upper else "OPEN"
        if self.interval_type != expected:
            raise ValueError(
                "equal endpoints require SINGLETON; distinct endpoints require OPEN"
            )
        return self


def _interval(lower, upper) -> RationalIsolatingInterval:  # type: ignore[no-untyped-def]
    return RationalIsolatingInterval(
        lower=_rational(lower),
        upper=_rational(upper),
        interval_type="SINGLETON" if lower == upper else "OPEN",
    )


def isolate_real_algebraic(value: RealAlgebraicValue) -> RationalIsolatingInterval:
    """Return SymPy's deterministic exact interval for the selected real root."""

    intervals = _sympy_polynomial(value).intervals()
    (lower, upper), _multiplicity = intervals[value.real_root_index]
    return _interval(lower, upper)


def _order_data(
    left: RealAlgebraicValue,
    right: RealAlgebraicValue,
) -> tuple[
    RealAlgebraicOrder,
    RationalIsolatingInterval,
    RationalIsolatingInterval,
]:
    left_poly = _sympy_polynomial(left)
    right_poly = _sympy_polynomial(right)
    if left.polynomial == right.polynomial:
        left_interval = isolate_real_algebraic(left)
        right_interval = isolate_real_algebraic(right)
        order: RealAlgebraicOrder = (
            "LT"
            if left.real_root_index < right.real_root_index
            else "GT"
            if left.real_root_index > right.real_root_index
            else "EQ"
        )
        return order, left_interval, right_interval

    # Distinct canonical minimal polynomials are coprime.  Isolating the roots
    # of their square-free product gives one exact common ordered axis, avoiding
    # floating-point matching between separately isolated root lists.
    product_intervals = (left_poly * right_poly).intervals()
    left_seen = right_seen = 0
    selected_left: tuple[int, RationalIsolatingInterval] | None = None
    selected_right: tuple[int, RationalIsolatingInterval] | None = None
    for position, ((lower, upper), _multiplicity) in enumerate(product_intervals):
        if _strict_root_count(left_poly, lower, upper):
            if left_seen == left.real_root_index:
                selected_left = (position, _interval(lower, upper))
            left_seen += 1
        if _strict_root_count(right_poly, lower, upper):
            if right_seen == right.real_root_index:
                selected_right = (position, _interval(lower, upper))
            right_seen += 1

    if selected_left is None or selected_right is None:  # pragma: no cover
        raise RuntimeError("exact real-root isolation lost a selected root")
    left_position, left_interval = selected_left
    right_position, right_interval = selected_right
    order = "LT" if left_position < right_position else "GT"
    return order, left_interval, right_interval


class RealAlgebraicOrderValue(StrictModel):
    """Source-bound exact order with rational root-isolation evidence."""

    left: RealAlgebraicValue
    right: RealAlgebraicValue
    order: RealAlgebraicOrder
    left_isolating_interval: RationalIsolatingInterval
    right_isolating_interval: RationalIsolatingInterval
    comparison_basis: Literal["ORDERED_REAL_ROOT_ISOLATION"] = (
        "ORDERED_REAL_ROOT_ISOLATION"
    )

    @model_validator(mode="after")
    def bind_exact_order(self) -> Self:
        order, left_interval, right_interval = _order_data(self.left, self.right)
        if self.order != order:
            raise ValueError("order must match the selected exact real roots")
        if self.left_isolating_interval != left_interval:
            raise ValueError("left isolating interval does not match the selected root")
        if self.right_isolating_interval != right_interval:
            raise ValueError(
                "right isolating interval does not match the selected root"
            )
        return self


def compare_real_algebraic(
    left: RealAlgebraicValue,
    right: RealAlgebraicValue,
) -> RealAlgebraicOrderValue:
    """Compare two bounded real algebraic values exactly."""

    order, left_interval, right_interval = _order_data(left, right)
    return RealAlgebraicOrderValue(
        left=left,
        right=right,
        order=order,
        left_isolating_interval=left_interval,
        right_isolating_interval=right_interval,
    )


__all__ = [
    "MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS",
    "MAX_REAL_ALGEBRAIC_DEGREE",
    "RationalIsolatingInterval",
    "RealAlgebraicOrderValue",
    "RealAlgebraicValue",
    "compare_real_algebraic",
    "isolate_real_algebraic",
]
