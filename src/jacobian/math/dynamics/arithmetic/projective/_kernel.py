"""Exact bounded projective dynamics on P1(Q)."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)

MAX_PROJECTIVE_DEGREE = 12
MAX_PROJECTIVE_STEPS = 256
# These are operation-envelope limits, narrower than the shared wire carrier.
MAX_PROJECTIVE_COMPONENT_DIGITS = 8_192
MAX_PROJECTIVE_SERIALIZED_DIGITS = 2_000_000
MAX_PROJECTIVE_CRITICAL_WORK = 200_000


class ProjectivePoint(StrictModel):
    x: CanonicalRational
    y: CanonicalRational

    @model_validator(mode="after")
    def normalize(self) -> Self:
        if self.x.as_fraction() == 0 and self.y.as_fraction() == 0:
            raise ValueError("projective point cannot be zero")
        if self.x.as_fraction() != 0:
            if self.x.as_fraction() != 1:
                raise ValueError(
                    "projective points must be normalized with x=1 when finite"
                )
        elif self.y.as_fraction() != 1:
            raise ValueError("infinity must be normalized as [0:1]")
        return self

    @classmethod
    def from_fraction(cls, value: Fraction) -> Self:
        # A finite point is [1:value] in the normalized homogeneous chart.
        return cls(
            x=CanonicalRational.from_integer_ratio(1, 1),
            y=CanonicalRational.from_fraction(value),
        )

    @classmethod
    def infinity(cls) -> Self:
        return cls(
            x=CanonicalRational.from_integer_ratio(0, 1),
            y=CanonicalRational.from_integer_ratio(1, 1),
        )

    def affine(self) -> Fraction | None:
        return (
            None
            if self.x.as_fraction() == 0
            else self.y.as_fraction() / self.x.as_fraction()
        )


class HomogeneousProjectiveMap(StrictModel):
    degree: int = Field(ge=0, le=MAX_PROJECTIVE_DEGREE)
    numerator: tuple[CanonicalRational, ...]
    denominator: tuple[CanonicalRational, ...]

    @model_validator(mode="after")
    def shape(self) -> Self:
        if (
            len(self.numerator) != self.degree + 1
            or len(self.denominator) != self.degree + 1
        ):
            raise ValueError(
                "homogeneous coefficient tuples must have degree plus one entries"
            )
        # A degree-zero constant map such as [0:1] is a valid projective map.
        if all(v.as_fraction() == 0 for v in self.numerator) and all(
            v.as_fraction() == 0 for v in self.denominator
        ):
            raise ValueError("homogeneous map coordinates cannot both vanish")
        return self

    def eval_homogeneous(self, point: ProjectivePoint) -> tuple[Fraction, Fraction]:
        x, y = point.x.as_fraction(), point.y.as_fraction()

        def ev(coeff: tuple[CanonicalRational, ...]) -> Fraction:
            return sum(
                (
                    c.as_fraction() * (x**i) * (y ** (self.degree - i))
                    for i, c in enumerate(coeff)
                ),
                Fraction(0),
            )

        return ev(self.numerator), ev(self.denominator)


def _digits(value: int) -> int:
    return len(format_canonical_integer(abs(value)))


def _height(value: CanonicalRational) -> tuple[int, int]:
    return _digits(value.num), _digits(value.den)


def _height_add(
    left: tuple[int, int] | None, right: tuple[int, int] | None
) -> tuple[int, int] | None:
    if left is None:
        return right
    if right is None:
        return left
    # A common denominator is a safe pre-arithmetic bound. Reduction can only
    # make the resulting components smaller.
    return (
        max(left[0] + right[1], right[0] + left[1]) + 1,
        left[1] + right[1],
    )


def _height_mul(
    left: tuple[int, int] | None, right: tuple[int, int] | None
) -> tuple[int, int] | None:
    if left is None or right is None:
        return None
    return left[0] + right[0], left[1] + right[1]


def _height_pow(value: tuple[int, int] | None, exponent: int) -> tuple[int, int] | None:
    if value is None:
        return None
    return value[0] * exponent, value[1] * exponent


def _check_height(
    value: tuple[int, int] | None, location: tuple[str | int, ...]
) -> None:
    if value is not None and max(value) > MAX_PROJECTIVE_COMPONENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=location,
            code="arithmetic_dynamics.projective_growth_bound",
            message="projective intermediate or output exceeds the component digit envelope",
        )


# The preflight uses binary logarithmic bounds, not decimal digit recurrences.
# For a nonzero integer a, ``a.bit_length()`` gives |a| < 2**bits.  Products
# therefore add bit bounds, while rational addition uses an explicit common
# denominator and one carry bit per addition.  This is a structure-aware bound:
# zero coefficients disappear. One-term maps (identity and z -> z**2) do not
# pay for absent terms. It is still a bound, rather than a second evaluation.
_LOG10_2_NUMERATOR = 30103
_LOG10_2_DENOMINATOR = 100000
_ZERO_BOUND = (0, 0)


def _bit_bound(value: CanonicalRational) -> tuple[int, int]:
    if value.num == 0:
        return _ZERO_BOUND
    return (abs(value.num).bit_length(), value.den.bit_length())


def _bound_add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    if left == _ZERO_BOUND:
        return right
    if right == _ZERO_BOUND:
        return left
    # Keep the two summands separate so the denominator product is explicit;
    # this remains valid even when their denominators are coprime.
    numerator = max(left[0] + right[1], right[0] + left[1]) + 1
    denominator = left[1] + right[1]
    return numerator, denominator


def _bound_mul(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    if left == _ZERO_BOUND or right == _ZERO_BOUND:
        return _ZERO_BOUND
    return left[0] + right[0], left[1] + right[1]


def _bound_pow(value: tuple[int, int], exponent: int) -> tuple[int, int]:
    if exponent == 0:
        return (1, 1)
    if value == _ZERO_BOUND:
        return _ZERO_BOUND
    return value[0] * exponent, value[1] * exponent


def _decimal_digits_from_bits(bits: int) -> int:
    if bits <= 1:
        return 1
    # log10(2) < 30103/100000.  The ceiling is taken using integers, so the
    # admission result does not depend on a platform floating-point libm.
    return (
        (bits - 1) * _LOG10_2_NUMERATOR + _LOG10_2_DENOMINATOR - 1
    ) // _LOG10_2_DENOMINATOR + 1


def _check_bit_bound(value: tuple[int, int], location: tuple[str | int, ...]) -> None:
    if (
        max(_decimal_digits_from_bits(bits) for bits in value)
        > MAX_PROJECTIVE_COMPONENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=location,
            code="arithmetic_dynamics.projective_growth_bound",
            message="projective intermediate or output exceeds the component digit envelope",
        )


def _bound_serialized_digits(value: tuple[int, int]) -> int:
    return sum(_decimal_digits_from_bits(bits) for bits in value)


def _serialized_point_digits(point: ProjectivePoint) -> int:
    return sum((*_height(point.x), *_height(point.y)))


def _serialized_map_digits(map_value: HomogeneousProjectiveMap) -> int:
    return sum(
        sum((*_height(value),))
        for value in (*map_value.numerator, *map_value.denominator)
    )


def _check_serialized_digits(digits: int, location: tuple[str | int, ...]) -> None:
    if digits > MAX_PROJECTIVE_SERIALIZED_DIGITS:
        raise OperationResourceAdmissionError(
            location=location,
            code="arithmetic_dynamics.projective_serialized_bound",
            message="projective result serialization exceeds the admitted envelope",
        )


def _require_map(
    value: object, location: tuple[str | int, ...]
) -> HomogeneousProjectiveMap:
    if not isinstance(value, HomogeneousProjectiveMap):
        raise OperationDomainValidationError(
            location=location,
            code="arithmetic_dynamics.projective_map_type",
            message="expected a homogeneous projective map",
        )
    if type(value.degree) is not int or not 0 <= value.degree <= MAX_PROJECTIVE_DEGREE:
        raise OperationDomainValidationError(
            location=(*location, "degree"),
            code="arithmetic_dynamics.projective_degree",
            message="projective degree must be an integer in the admitted range",
        )
    if (
        len(value.numerator) != value.degree + 1
        or len(value.denominator) != value.degree + 1
    ):
        raise OperationDomainValidationError(
            location=location,
            code="arithmetic_dynamics.projective_map_shape",
            message="homogeneous coefficient tuples have the wrong shape",
        )
    if any(
        not isinstance(coefficient, CanonicalRational)
        for coefficient in (*value.numerator, *value.denominator)
    ):
        raise OperationDomainValidationError(
            location=location,
            code="arithmetic_dynamics.projective_coefficient_type",
            message="projective coefficients must be canonical rationals",
        )
    for coefficient in (*value.numerator, *value.denominator):
        _check_height(_height(coefficient), (*location, "coefficients"))
    if all(coefficient.num == 0 for coefficient in value.numerator) and all(
        coefficient.num == 0 for coefficient in value.denominator
    ):
        raise OperationDomainValidationError(
            location=location,
            code="arithmetic_dynamics.projective_zero_map",
            message="homogeneous map coordinates cannot both vanish",
        )
    return value


def _require_point(value: object, location: tuple[str | int, ...]) -> ProjectivePoint:
    if not isinstance(value, ProjectivePoint):
        raise OperationDomainValidationError(
            location=location,
            code="arithmetic_dynamics.projective_point_type",
            message="expected a normalized projective point",
        )
    if not isinstance(value.x, CanonicalRational) or not isinstance(
        value.y, CanonicalRational
    ):
        raise OperationDomainValidationError(
            location=location,
            code="arithmetic_dynamics.projective_point_shape",
            message="projective coordinates must be canonical rationals",
        )
    _check_height(_height(value.x), (*location, "x"))
    _check_height(_height(value.y), (*location, "y"))
    if value.x.num == 0:
        valid = value.y.num == 1 and value.y.den == 1
    else:
        valid = value.x.num == 1 and value.x.den == 1
    if not valid or (value.x.num == 0 and value.y.num == 0):
        raise OperationDomainValidationError(
            location=location,
            code="arithmetic_dynamics.projective_point_shape",
            message="projective point is not in normalized form",
        )
    return value


def _eval_height_bounds(
    map_value: HomogeneousProjectiveMap,
    point_bounds: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    x_bound, y_bound = point_bounds

    def evaluate(coefficients: tuple[CanonicalRational, ...]) -> tuple[int, int]:
        result = _ZERO_BOUND
        for index, coefficient in enumerate(coefficients):
            term = _bound_mul(
                _bound_mul(
                    _bit_bound(coefficient),
                    _bound_pow(x_bound, index),
                ),
                _bound_pow(y_bound, map_value.degree - index),
            )
            result = _bound_add(result, term)
        return result

    return evaluate(map_value.numerator), evaluate(map_value.denominator)


def _normalize_height_bounds(
    numerator: tuple[int, int], denominator: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    # The public point is normalized after evaluation.  A zero coordinate or a
    # zero denominator produces infinity; otherwise [F:G] has affine value
    # F/G, whose numerator and denominator are bounded by cross-products.
    if numerator == _ZERO_BOUND or denominator == _ZERO_BOUND:
        return _ZERO_BOUND, (1, 1)
    return (1, 1), (numerator[0] + denominator[1], numerator[1] + denominator[0])


def _preflight_apply(
    map_value: HomogeneousProjectiveMap,
    point: ProjectivePoint,
    *,
    location: tuple[str | int, ...] = (),
) -> tuple[tuple[int, int], tuple[int, int]]:
    bounds = _eval_height_bounds(map_value, (_bit_bound(point.x), _bit_bound(point.y)))
    for value in bounds:
        _check_bit_bound(value, (*location, "image"))
    normalized = _normalize_height_bounds(*bounds)
    for value in normalized:
        _check_bit_bound(value, (*location, "image"))
    return normalized


def _point(a: Fraction, b: Fraction) -> ProjectivePoint:
    if a == 0 and b == 0:
        raise OperationDomainValidationError(
            location=("point",),
            code="arithmetic_dynamics.projective_base_locus",
            message="map is undefined at this point",
        )
    if a == 0:
        return ProjectivePoint.infinity()
    return ProjectivePoint.from_fraction(b / a)


def apply_projective_map(
    map_value: HomogeneousProjectiveMap, point: ProjectivePoint
) -> ProjectivePoint:
    map_value = _require_map(map_value, ("map",))
    point = _require_point(point, ("point",))
    predicted = _preflight_apply(map_value, point)
    _check_serialized_digits(
        _serialized_map_digits(map_value)
        + _serialized_point_digits(point)
        + sum(predicted[0] + predicted[1]),
        ("result",),
    )
    a, b = map_value.eval_homogeneous(point)
    return _point(a, b)


def _poly_height_add(
    left: tuple[tuple[int, int] | None, ...],
    right: tuple[tuple[int, int] | None, ...],
) -> tuple[tuple[int, int] | None, ...]:
    size = max(len(left), len(right))
    return tuple(
        _checked_poly_height_add(
            left[index] if index < len(left) else None,
            right[index] if index < len(right) else None,
        )
        for index in range(size)
    )


def _checked_poly_height_add(
    left: tuple[int, int] | None, right: tuple[int, int] | None
) -> tuple[int, int] | None:
    result = _height_add(left, right)
    _check_height(result, ("composition",))
    return result


def _poly_height_mul(
    left: tuple[tuple[int, int] | None, ...],
    right: tuple[tuple[int, int] | None, ...],
) -> tuple[tuple[int, int] | None, ...]:
    result: list[tuple[int, int] | None] = [None] * (len(left) + len(right) - 1)
    for i, first in enumerate(left):
        for j, second in enumerate(right):
            result[i + j] = _checked_poly_height_add(
                result[i + j], _height_mul(first, second)
            )
    return tuple(result)


def _poly_height_pow(
    value: tuple[tuple[int, int] | None, ...], exponent: int
) -> tuple[tuple[int, int] | None, ...]:
    result: tuple[tuple[int, int] | None, ...] = ((1, 1),)
    for _ in range(exponent):
        result = _poly_height_mul(result, value)
    return result


def _preflight_composition(
    outer: HomogeneousProjectiveMap, inner: HomogeneousProjectiveMap
) -> tuple[tuple[int, int] | None, ...]:
    inner_n = tuple(_height(value) for value in inner.numerator)
    inner_d = tuple(_height(value) for value in inner.denominator)

    def substitute(
        coefficients: tuple[CanonicalRational, ...],
    ) -> tuple[tuple[int, int] | None, ...]:
        result: tuple[tuple[int, int] | None, ...] = (None,) * (
            outer.degree * inner.degree + 1
        )
        for index, coefficient in enumerate(coefficients):
            term = _poly_height_mul(
                _poly_height_pow(inner_n, index),
                _poly_height_pow(inner_d, outer.degree - index),
            )
            term = tuple(_height_mul(_height(coefficient), value) for value in term)
            result = _poly_height_add(result, term)
        return result

    numerator, denominator = substitute(outer.numerator), substitute(outer.denominator)
    for value in (*numerator, *denominator):
        _check_height(value, ("composition",))
    return (*numerator, *denominator)


def compose_projective_maps(
    outer: HomogeneousProjectiveMap, inner: HomogeneousProjectiveMap
) -> HomogeneousProjectiveMap:
    outer = _require_map(outer, ("outer",))
    inner = _require_map(inner, ("inner",))
    degree = outer.degree * inner.degree
    if degree > MAX_PROJECTIVE_DEGREE:
        raise OperationResourceAdmissionError(
            location=("outer", "inner"),
            code="arithmetic_dynamics.projective_degree_bound",
            message="composition degree exceeds bound",
        )
    predicted = _preflight_composition(outer, inner)
    _check_serialized_digits(
        _serialized_map_digits(outer)
        + _serialized_map_digits(inner)
        + sum(value[0] + value[1] for value in predicted if value is not None),
        ("composition",),
    )

    def mul(a: Sequence[Fraction], b: Sequence[Fraction]) -> list[Fraction]:
        out = [Fraction(0)] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                out[i + j] += ai * bj
        return out

    inner_numerator = tuple(value.as_fraction() for value in inner.numerator)
    inner_denominator = tuple(value.as_fraction() for value in inner.denominator)

    def powers(a: Sequence[Fraction]) -> tuple[list[Fraction], ...]:
        result = [[Fraction(1)]]
        for _ in range(outer.degree):
            result.append(mul(result[-1], a))
        return tuple(result)

    numerator_powers = powers(inner_numerator)
    denominator_powers = powers(inner_denominator)

    def subst(coeff: tuple[CanonicalRational, ...]) -> tuple[CanonicalRational, ...]:
        out = [Fraction(0)] * (degree + 1)
        for i, c in enumerate(coeff):
            term = mul(numerator_powers[i], denominator_powers[outer.degree - i])
            for j, v in enumerate(term):
                out[j] += c.as_fraction() * v
        return tuple(CanonicalRational.from_fraction(v) for v in out)

    numerator = subst(outer.numerator)
    denominator = subst(outer.denominator)
    if all(value.num == 0 for value in numerator) and all(
        value.num == 0 for value in denominator
    ):
        raise OperationDomainValidationError(
            location=("composition",),
            code="arithmetic_dynamics.projective_base_locus",
            message="composition is undefined because both homogeneous coordinates vanish",
        )
    return HomogeneousProjectiveMap(
        degree=degree,
        numerator=numerator,
        denominator=denominator,
    )


def _derivative_height_bounds(
    coefficients: tuple[CanonicalRational, ...],
    point_bounds: tuple[tuple[int, int], tuple[int, int]],
    degree: int,
    *,
    with_respect_to_x: bool,
) -> tuple[int, int]:
    x_bound, y_bound = point_bounds
    result = _ZERO_BOUND
    for index, coefficient in enumerate(coefficients):
        factor = index if with_respect_to_x else degree - index
        if factor == 0 or coefficient.num == 0:
            continue
        x_exponent = index - 1 if with_respect_to_x else index
        y_exponent = degree - index if with_respect_to_x else degree - index - 1
        x_power = _bound_pow(x_bound, x_exponent)
        y_power = _bound_pow(y_bound, y_exponent)
        _check_bit_bound(x_power, ("critical", "derivative", "power"))
        _check_bit_bound(y_power, ("critical", "derivative", "power"))
        term = _bound_mul(
            _bound_mul(_bit_bound(coefficient), (factor.bit_length(), 1)),
            _bound_mul(x_power, y_power),
        )
        _check_bit_bound(term, ("critical", "derivative", "term"))
        result = _bound_add(result, term)
        _check_bit_bound(result, ("critical", "derivative", "sum"))
    return result


def _preflight_critical_point(
    map_value: HomogeneousProjectiveMap, point: ProjectivePoint
) -> None:
    point_bounds = (_bit_bound(point.x), _bit_bound(point.y))
    dx_n = _derivative_height_bounds(
        map_value.numerator, point_bounds, map_value.degree, with_respect_to_x=True
    )
    dx_d = _derivative_height_bounds(
        map_value.denominator, point_bounds, map_value.degree, with_respect_to_x=True
    )
    dy_n = _derivative_height_bounds(
        map_value.numerator, point_bounds, map_value.degree, with_respect_to_x=False
    )
    dy_d = _derivative_height_bounds(
        map_value.denominator, point_bounds, map_value.degree, with_respect_to_x=False
    )
    left = _bound_mul(dx_n, dy_d)
    right = _bound_mul(dy_n, dx_d)
    _check_bit_bound(left, ("critical", "determinant", "product"))
    _check_bit_bound(right, ("critical", "determinant", "product"))
    _check_bit_bound(_bound_add(left, right), ("critical", "determinant"))


def is_critical_point(
    map_value: HomogeneousProjectiveMap, point: ProjectivePoint
) -> bool:
    map_value = _require_map(map_value, ("map",))
    point = _require_point(point, ("point",))
    if map_value.degree < 2:
        return False
    _preflight_apply(map_value, point, location=("critical",))
    _preflight_critical_point(map_value, point)
    image_numerator, image_denominator = map_value.eval_homogeneous(point)
    if image_numerator == 0 and image_denominator == 0:
        raise OperationDomainValidationError(
            location=("point",),
            code="arithmetic_dynamics.projective_base_locus",
            message="criticality is undefined at a presentation base-locus point",
        )
    x, y = point.x.as_fraction(), point.y.as_fraction()

    def dx(coefficients: tuple[CanonicalRational, ...]) -> Fraction:
        return sum(
            (
                i * c.as_fraction() * x ** (i - 1) * y ** (map_value.degree - i)
                for i, c in enumerate(coefficients)
                if i and c.num != 0
            ),
            Fraction(0),
        )

    def dy(coefficients: tuple[CanonicalRational, ...]) -> Fraction:
        return sum(
            (
                (map_value.degree - i)
                * c.as_fraction()
                * x**i
                * y ** (map_value.degree - i - 1)
                for i, c in enumerate(coefficients)
                if i < map_value.degree and c.num != 0
            ),
            Fraction(0),
        )

    return (
        dx(map_value.numerator) * dy(map_value.denominator)
        - dy(map_value.numerator) * dx(map_value.denominator)
        == 0
    )


def _preflight_critical_orbits(
    map_value: HomogeneousProjectiveMap,
    points: Sequence[ProjectivePoint],
    max_steps: int,
) -> None:
    work = len(points) * max_steps * (map_value.degree + 1)
    if work > MAX_PROJECTIVE_CRITICAL_WORK:
        raise OperationResourceAdmissionError(
            location=("critical_points", "max_steps"),
            code="arithmetic_dynamics.projective_critical_work_bound",
            message="critical-orbit batch exceeds the admitted work envelope",
        )
    serialized = _serialized_map_digits(map_value)
    for index, point in enumerate(points):
        _require_point(point, ("critical_points", index))
        current = (_bit_bound(point.x), _bit_bound(point.y))
        # Each row retains the supplied point and the orbit tuple retains it
        # again as its zero-th entry.
        serialized += 2 * _serialized_point_digits(point)
        for _ in range(max_steps):
            raw = _eval_height_bounds(map_value, current)
            for value in raw:
                _check_bit_bound(value, ("critical", index, "evaluation"))
            current = _normalize_height_bounds(*raw)
            for value in current:
                _check_bit_bound(value, ("critical", index, "image"))
            serialized += _bound_serialized_digits(current[0])
            serialized += _bound_serialized_digits(current[1])
    _check_serialized_digits(serialized, ("critical",))


def _preflight_orbit(
    map_value: HomogeneousProjectiveMap, start: ProjectivePoint, max_steps: int
) -> None:
    current = (_bit_bound(start.x), _bit_bound(start.y))
    serialized = _serialized_map_digits(map_value) + _serialized_point_digits(start)
    for _ in range(max_steps):
        raw = _eval_height_bounds(map_value, current)
        for value in raw:
            _check_bit_bound(value, ("orbit", "evaluation"))
        current = _normalize_height_bounds(*raw)
        for value in current:
            _check_bit_bound(value, ("orbit", "image"))
        serialized += sum(_bound_serialized_digits(value) for value in current)
    _check_serialized_digits(serialized, ("orbit",))


def projective_orbit(
    map_value: HomogeneousProjectiveMap,
    start: ProjectivePoint,
    max_steps: int,
) -> tuple[tuple[ProjectivePoint, ...], tuple[int, int, int] | None]:
    map_value = _require_map(map_value, ("map",))
    start = _require_point(start, ("start",))
    if type(max_steps) is not int:
        raise OperationDomainValidationError(
            location=("max_steps",),
            code="arithmetic_dynamics.projective_steps_type",
            message="projective orbit steps must be an integer",
        )
    if max_steps < 0 or max_steps > MAX_PROJECTIVE_STEPS:
        raise OperationResourceAdmissionError(
            location=("max_steps",),
            code="arithmetic_dynamics.projective_steps_bound",
            message="projective orbit exceeds admitted bound",
        )
    _preflight_orbit(map_value, start, max_steps)
    orbit = [start]
    seen = {start: 0}
    for index in range(1, max_steps + 1):
        point = apply_projective_map(map_value, orbit[-1])
        orbit.append(point)
        if point in seen:
            return tuple(orbit), (seen[point], index, index - seen[point])
        seen[point] = index
    return tuple(orbit), None


class ProjectiveOrbitResult(StrictModel):
    map: HomogeneousProjectiveMap
    start: ProjectivePoint
    orbit: tuple[ProjectivePoint, ...]
    max_steps: int
    termination: Literal["REPEAT_FOUND", "STEP_BOUND_REACHED"]
    first_seen_index: int | None = None
    period: int | None = None
    exact_period: bool = False

    @classmethod
    def from_kernel(cls, **kwargs: Any) -> Self:
        return cls.model_construct(**kwargs)


__all__ = [
    "MAX_PROJECTIVE_COMPONENT_DIGITS",
    "MAX_PROJECTIVE_CRITICAL_WORK",
    "MAX_PROJECTIVE_DEGREE",
    "MAX_PROJECTIVE_SERIALIZED_DIGITS",
    "MAX_PROJECTIVE_STEPS",
    "HomogeneousProjectiveMap",
    "ProjectiveOrbitResult",
    "ProjectivePoint",
    "apply_projective_map",
    "compose_projective_maps",
    "is_critical_point",
    "projective_orbit",
]
