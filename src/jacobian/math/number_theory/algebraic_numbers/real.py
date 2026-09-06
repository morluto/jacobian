"""Canonical bounded real algebraic values and exact order."""

from __future__ import annotations

from collections.abc import Mapping
from math import gcd
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import Field, StrictInt, ValidateAs, WithJsonSchema, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math._root_isolation import strict_root_count

if TYPE_CHECKING:
    from sympy import Poly
    from sympy.core.numbers import Rational as SympyRational

# A degree-sixteen carrier includes coordinate projections of isolated
# intersections of plane quartics. Pairwise comparison retains its proven
# degree-eight envelope: the product of two such defining polynomials has
# degree at most sixteen and coefficient height below 2,002 decimal digits.
# Mignotte's root-separation bound then needs fewer than 32,768 decimal digits
# for rational isolating endpoints, so every accepted comparison remains
# representable by the shared canonical scalar envelope.
MAX_REAL_ALGEBRAIC_DEGREE = 16
MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE = 8
MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS = 1_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"real_algebraic.{reason}", message)


RealAlgebraicOrder = Literal["LT", "EQ", "GT"]


def _sympy_polynomial(value: RealAlgebraicValue) -> Poly:
    import sympy

    x = sympy.Symbol("x")
    return sympy.Poly.from_list(
        [parse_canonical_integer(coefficient) for coefficient in value.polynomial],
        gens=x,
        domain=sympy.ZZ,
    )


def _rational(value: SympyRational) -> CanonicalRational:
    import sympy

    rational = sympy.Rational(value)
    return CanonicalRational(
        num=format_canonical_integer(int(rational.p)),
        den=format_canonical_integer(int(rational.q)),
    )


class _RealAlgebraicValueShape(StrictModel):
    """Canonical structural representation of an indexed real root.

    ``polynomial`` is the primitive irreducible polynomial in ``ZZ[x]`` with
    positive leading coefficient, listed in descending degree.  The
    zero-based ``real_root_index`` selects one of its real roots in increasing
    order.  This pair uniquely determines the value and its real embedding.
    """

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
    def require_canonical_polynomial_shape(self) -> Self:
        if any(
            len(coefficient.lstrip("-")) > MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS
            for coefficient in self.polynomial
        ):
            raise _validation_error(
                "coefficient_bound",
                "real algebraic polynomial coefficients exceed the "
                f"{MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS}-digit bound",
            )
        coefficients = tuple(
            parse_canonical_integer(coefficient) for coefficient in self.polynomial
        )
        if coefficients[0] <= 0:
            raise _validation_error(
                "leading_sign",
                "real algebraic minimal polynomial must have positive leading coefficient",
            )
        content = 0
        for coefficient in coefficients:
            content = gcd(content, abs(coefficient))
        if content != 1:
            raise _validation_error(
                "not_primitive",
                "real algebraic minimal polynomial must be primitive over ZZ",
            )

        if self.real_root_index >= len(self.polynomial) - 1:
            raise _validation_error(
                "root_index",
                "real_root_index must be smaller than the polynomial degree",
            )
        return self


class RealAlgebraicValue(_RealAlgebraicValueShape):
    """One real algebraic number in canonical minimal-polynomial form.

    Parsing checks the canonical encoding. Consumers establish irreducibility
    and the selected real root within their admitted computation.
    """

    @classmethod
    def _from_admitted_polynomial(
        cls,
        *,
        polynomial: tuple[CanonicalInteger, ...],
        real_root_index: int,
    ) -> RealAlgebraicValue:
        """Construct after an owner has admitted the canonical polynomial/root."""

        return cls.model_construct(
            polynomial=polynomial,
            real_root_index=real_root_index,
        )


def _unrecognized_real_value_from_shape(
    shape: _RealAlgebraicValueShape,
) -> RealAlgebraicValue:
    if isinstance(shape, RealAlgebraicValue):
        return shape
    return RealAlgebraicValue.model_construct(
        polynomial=shape.polynomial,
        real_root_index=shape.real_root_index,
    )


def _number_field_embedding_root_schema() -> dict[str, object]:
    schema = RealAlgebraicValue.model_json_schema()
    polynomial = schema.get("properties", {}).get("polynomial")
    if isinstance(polynomial, dict):
        polynomial["maxItems"] = MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE + 1
    return schema


_UnrecognizedRealAlgebraicValue = Annotated[
    RealAlgebraicValue,
    ValidateAs(_RealAlgebraicValueShape, _unrecognized_real_value_from_shape),
    WithJsonSchema(_number_field_embedding_root_schema()),
]


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
            raise _validation_error(
                "interval_order",
                "isolating interval lower endpoint must not exceed upper",
            )
        expected = "SINGLETON" if lower == upper else "OPEN"
        if self.interval_type != expected:
            raise _validation_error(
                "interval_type",
                "equal endpoints require SINGLETON; distinct endpoints require OPEN",
            )
        return self


def _interval(lower: SympyRational, upper: SympyRational) -> RationalIsolatingInterval:
    return RationalIsolatingInterval(
        lower=_rational(lower),
        upper=_rational(upper),
        interval_type="SINGLETON" if lower == upper else "OPEN",
    )


def _admit_real_polynomial(value: RealAlgebraicValue) -> Poly:
    polynomial = _sympy_polynomial(value)
    if polynomial.is_irreducible is not True:
        raise OperationDomainValidationError(
            location=("value",),
            code="real_algebraic.not_irreducible",
            message="real algebraic minimal polynomial must be irreducible over QQ",
        )
    return polynomial


def isolate_real_algebraic(value: RealAlgebraicValue) -> RationalIsolatingInterval:
    """Return an exact interval after admitting the selected real root."""

    intervals = _admit_real_polynomial(value).intervals()
    if value.real_root_index >= len(intervals):
        raise OperationDomainValidationError(
            location=("value", "real_root_index"),
            code="real_algebraic.root_index",
            message="real_root_index must select an existing real root",
        )
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
    if left.polynomial == right.polynomial:
        intervals = _admit_real_polynomial(left).intervals()
        if max(left.real_root_index, right.real_root_index) >= len(intervals):
            raise OperationDomainValidationError(
                location=("left", "right"),
                code="real_algebraic.root_index",
                message="real_root_index must select an existing real root",
            )
        left_interval = _interval(*intervals[left.real_root_index][0])
        right_interval = _interval(*intervals[right.real_root_index][0])
        order: RealAlgebraicOrder = (
            "LT"
            if left.real_root_index < right.real_root_index
            else "GT"
            if left.real_root_index > right.real_root_index
            else "EQ"
        )
        return order, left_interval, right_interval

    if any(
        len(value.polynomial) - 1 > MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE
        for value in (left, right)
    ):
        raise ValueError(
            "exact algebraic comparison admits degree at most "
            f"{MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE}"
        )

    # Distinct canonical minimal polynomials are coprime.  Isolating the roots
    # of their square-free product gives one exact common ordered axis, avoiding
    # floating-point matching between separately isolated root lists.
    left_poly = _admit_real_polynomial(left)
    right_poly = _admit_real_polynomial(right)
    product_intervals = (left_poly * right_poly).intervals()
    left_seen = right_seen = 0
    selected_left: tuple[int, RationalIsolatingInterval] | None = None
    selected_right: tuple[int, RationalIsolatingInterval] | None = None
    for position, ((lower, upper), _multiplicity) in enumerate(product_intervals):
        if selected_left is None and strict_root_count(left_poly, lower, upper):
            if left_seen == left.real_root_index:
                selected_left = (position, _interval(lower, upper))
            left_seen += 1
        if selected_right is None and strict_root_count(right_poly, lower, upper):
            if right_seen == right.real_root_index:
                selected_right = (position, _interval(lower, upper))
            right_seen += 1
        if selected_left is not None and selected_right is not None:
            break

    if selected_left is None or selected_right is None:
        raise OperationDomainValidationError(
            location=("left", "right"),
            code="real_algebraic.root_index",
            message="real_root_index must select an existing real root",
        )
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

    @model_validator(mode="before")
    @classmethod
    def require_raw_comparison_degree_bound(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        left = data.get("left")
        right = data.get("right")
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return data
        left_polynomial = left.get("polynomial")
        right_polynomial = right.get("polynomial")
        if (
            isinstance(left_polynomial, (list, tuple))
            and isinstance(right_polynomial, (list, tuple))
            and left_polynomial != right_polynomial
            and any(
                len(polynomial) - 1 > MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE
                for polynomial in (left_polynomial, right_polynomial)
            )
        ):
            raise _validation_error(
                "comparison_degree_bound",
                "exact algebraic comparison admits degree at most "
                f"{MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE} for distinct polynomials",
            )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_admitted_distinct_polynomial_pair(self) -> Self:
        if self.left.polynomial != self.right.polynomial and any(
            len(value.polynomial) - 1 > MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE
            for value in (self.left, self.right)
        ):
            raise _validation_error(
                "comparison_degree_bound",
                "exact algebraic comparison admits degree at most "
                f"{MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE} for distinct polynomials",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        left: RealAlgebraicValue,
        right: RealAlgebraicValue,
        order: RealAlgebraicOrder,
        left_isolating_interval: RationalIsolatingInterval,
        right_isolating_interval: RationalIsolatingInterval,
    ) -> Self:
        """Construct after the exact common-axis kernel established the order."""

        return cls.model_construct(
            left=left,
            right=right,
            order=order,
            left_isolating_interval=left_isolating_interval,
            right_isolating_interval=right_isolating_interval,
        )


def compare_real_algebraic(
    left: RealAlgebraicValue,
    right: RealAlgebraicValue,
) -> RealAlgebraicOrderValue:
    """Compare two bounded real algebraic values exactly."""

    order, left_interval, right_interval = _order_data(left, right)
    return RealAlgebraicOrderValue._from_kernel(
        left=left,
        right=right,
        order=order,
        left_isolating_interval=left_interval,
        right_isolating_interval=right_interval,
    )


def _compare_admitted_real_algebraic(
    left: RealAlgebraicValue,
    right: RealAlgebraicValue,
    left_interval: RationalIsolatingInterval,
    right_interval: RationalIsolatingInterval,
) -> RealAlgebraicOrder:
    """Order roots whose irreducibility and intervals were admitted upstream."""
    if left.polynomial == right.polynomial:
        if left.real_root_index == right.real_root_index:
            return "EQ"
        return "LT" if left.real_root_index < right.real_root_index else "GT"
    left_lower = left_interval.lower.as_fraction()
    left_upper = left_interval.upper.as_fraction()
    right_lower = right_interval.lower.as_fraction()
    right_upper = right_interval.upper.as_fraction()
    if left_upper <= right_lower:
        return "LT"
    if right_upper <= left_lower:
        return "GT"
    left_poly = _sympy_polynomial(left)
    right_poly = _sympy_polynomial(right)
    left_position = right_position = None
    left_seen = right_seen = 0
    for position, ((lower, upper), _multiplicity) in enumerate(
        (left_poly * right_poly).intervals()
    ):
        if left_position is None and strict_root_count(left_poly, lower, upper):
            if left_seen == left.real_root_index:
                left_position = position
            left_seen += 1
        if right_position is None and strict_root_count(right_poly, lower, upper):
            if right_seen == right.real_root_index:
                right_position = position
            right_seen += 1
        if left_position is not None and right_position is not None:
            return "LT" if left_position < right_position else "GT"
    raise ValueError("admitted root intervals do not establish an order")


__all__ = [
    "MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS",
    "MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE",
    "MAX_REAL_ALGEBRAIC_DEGREE",
    "RationalIsolatingInterval",
    "RealAlgebraicOrderValue",
    "RealAlgebraicValue",
    "compare_real_algebraic",
    "isolate_real_algebraic",
]
