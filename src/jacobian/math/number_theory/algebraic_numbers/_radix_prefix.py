"""Exact bounded radix prefixes of canonical real algebraic values (#2789).

The prefix is derived by exact comparison only.  Rational values use exact
integer division under the canonical terminating-zero convention; irrational
values are handled by isolating the root of the scaled polynomial
``b**(q*d) * f(x / b**q)`` for degree ``d``, whose selected real root is
exactly ``b**q * alpha``,
so the integer part of the scaled value is obtained from one exact isolating
interval.  No binary floating point enters the result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import DecimalIntegerEncoding
from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_execution,
)
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.algebraic_numbers._radix_prefix_process import (
    run_scaled_integer_part_worker,
)
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS,
    RealAlgebraicValue,
    _admit_real_polynomial,
)

MAX_RADIX_BASE = 36
MAX_RADIX_PLACES = 256
# The transformed polynomial is private execution state, but its coefficient
# height and the root-isolation precision still need an explicit envelope
# before SymPy sees it.  These limits are deliberately larger than the worst
# case induced by the public source (degree 16, 1,000 coefficient digits,
# base 36 and 256 places), while preventing an accidental future widening of
# the public fields from turning into an unbounded backend request.
MAX_RADIX_SCALED_COEFFICIENT_DIGITS = 16_384
MAX_RADIX_ISOLATION_BITS = 1_048_576
# Sum of retained exact-integer digits, digit entries, and fixed scalar slots.
# Concrete transports enforce their own independent encoded-byte ceilings.
MAX_RADIX_RESULT_ALLOCATION_UNITS = 32_768
# Cauchy's bound on a degree-d integer polynomial of 1,000-digit coefficients
# can place the floor one digit past the coefficient envelope, e.g. floor of
# the negative root of x^2 + (10**1000-1)x - (10**1000-1) is -10**1000.
MAX_RADIX_INTEGER_PART_DIGITS = MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS + 1
RadixIntegerPart = Annotated[
    int, DecimalIntegerEncoding(max_digits=MAX_RADIX_INTEGER_PART_DIGITS)
]


def _unique_floor_of_open_interval(lower: Any, upper: Any) -> int | None:
    """Return the only possible floor in an open rational interval."""

    floor_lower = lower.numerator // lower.denominator
    ceil_upper = -((-upper.numerator) // upper.denominator)
    greatest_integer_below_upper = ceil_upper - 1
    return int(floor_lower) if floor_lower == greatest_integer_below_upper else None


class RadixPrefixRequest(StrictModel):
    """Request an exact base-b prefix of one canonical real algebraic value."""

    value: RealAlgebraicValue
    base: StrictInt = Field(
        ge=2,
        le=MAX_RADIX_BASE,
        description=f"Integer radix in the supported range [2, {MAX_RADIX_BASE}].",
    )
    fractional_places: StrictInt = Field(
        ge=0,
        description=(
            "Number of fractional digits requested; the bounded exact "
            f"operation admits at most {MAX_RADIX_PLACES}."
        ),
    )


class RadixPrefixResult(StrictModel):
    """An exact source-bound base-b prefix with an explicit convention."""

    value: RealAlgebraicValue
    base: StrictInt = Field(ge=2, le=MAX_RADIX_BASE)
    fractional_places: StrictInt = Field(ge=0, le=MAX_RADIX_PLACES)
    integer_part: RadixIntegerPart
    fractional_digits: tuple[StrictInt, ...] = Field(
        max_length=MAX_RADIX_PLACES,
        description="Exactly one digit in [0, base) for each requested place.",
    )
    convention: Literal["TERMINATING_ZEROS_FOR_RATIONALS"] = (
        "TERMINATING_ZEROS_FOR_RATIONALS"
    )

    @model_validator(mode="after")
    def require_digit_shape(self) -> Self:
        if len(self.fractional_digits) != self.fractional_places:
            raise PydanticCustomError(
                "algebraic_number.radix_digit_count",
                "fractional digit count must equal fractional_places",
            )
        if any(digit < 0 or digit >= self.base for digit in self.fractional_digits):
            raise PydanticCustomError(
                "algebraic_number.radix_digit_range",
                "every fractional digit must lie in [0, base)",
            )
        return self


@dataclass(frozen=True, slots=True)
class _RadixAdmission:
    """Request-scoped arithmetic envelope for one prefix."""

    scale: int
    isolation_bits: int


def _require_request(base: int, fractional_places: int) -> None:
    if base < 2 or base > MAX_RADIX_BASE:
        raise OperationDomainValidationError(
            location=("base",),
            code="algebraic_number.radix_base_out_of_range",
            message=f"base must lie in [2, {MAX_RADIX_BASE}]",
        )
    if fractional_places < 0:
        raise OperationDomainValidationError(
            location=("fractional_places",),
            code="algebraic_number.radix_places_out_of_range",
            message="fractional_places must be nonnegative",
        )
    if fractional_places > MAX_RADIX_PLACES:
        raise OperationResourceAdmissionError(
            location=("fractional_places",),
            code="algebraic_number.radix_places_bound",
            message=(
                f"fractional_places exceeds the admitted bound {MAX_RADIX_PLACES}; "
                "the scaled defining polynomial would exceed the coefficient envelope"
            ),
        )


def _admit_request(request: RadixPrefixRequest) -> _RadixAdmission:
    """Reserve transformed-polynomial, isolation, and result resources.

    This is intentionally structural: it does not call SymPy or establish a
    root identity.  The domain admission below performs that one semantic
    check, and the kernel then reuses the admitted source unchanged.
    """

    _require_request(request.base, request.fractional_places)
    degree = len(request.value.polynomial) - 1
    scale = request.base**request.fractional_places
    scale_digits = len(format_canonical_integer(scale))
    scaled_coefficient_digits = max(
        len(format_canonical_integer(abs(coefficient))) + position * scale_digits
        for position, coefficient in enumerate(request.value.polynomial)
    )
    if scaled_coefficient_digits > MAX_RADIX_SCALED_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("value", "polynomial"),
            code="algebraic_number.radix_scaled_coefficient_bound",
            message=(
                "the scaled defining polynomial exceeds the admitted "
                f"{MAX_RADIX_SCALED_COEFFICIENT_DIGITS}-digit coefficient envelope"
            ),
        )
    # A root of an integer polynomial is separated from an integer by a
    # bound exponential in coefficient bit height and degree.  Reserve a
    # conservative bit envelope before constructing the scaled polynomial.
    coefficient_bits = (scaled_coefficient_digits * 100_000 + 30_102) // 30_103
    isolation_bits = degree * (coefficient_bits + degree.bit_length() + 8) + 8
    if isolation_bits > MAX_RADIX_ISOLATION_BITS:
        raise OperationResourceAdmissionError(
            location=("value",),
            code="algebraic_number.radix_isolation_bound",
            message=(
                "root isolation for the scaled value exceeds the admitted "
                f"{MAX_RADIX_ISOLATION_BITS}-bit precision envelope"
            ),
        )

    # The source is retained in the result.  Reserve its exact integer digits,
    # the bounded integer part, every radix digit, and fixed scalar slots.  This
    # is independent of JSON spelling or any other delivery format.
    source_digit_units = sum(
        len(format_canonical_integer(abs(coefficient)))
        for coefficient in request.value.polynomial
    )
    integer_part_digits = (
        max(
            len(format_canonical_integer(abs(coefficient)))
            for coefficient in request.value.polynomial
        )
        + 2
    )
    allocation_units = (
        source_digit_units
        + len(request.value.polynomial)
        + integer_part_digits
        + request.fractional_places
        + 8
    )
    if allocation_units > MAX_RADIX_RESULT_ALLOCATION_UNITS:
        raise OperationResourceAdmissionError(
            location=("fractional_places",),
            code="algebraic_number.radix_result_allocation_bound",
            message=(
                "the exact radix prefix exceeds the admitted "
                f"{MAX_RADIX_RESULT_ALLOCATION_UNITS}-unit result allocation"
            ),
        )
    return _RadixAdmission(
        scale=scale,
        isolation_bits=isolation_bits,
    )


def _rational_value(value: RealAlgebraicValue) -> Fraction | None:
    """Return the exact rational the value denotes, if the root is rational."""

    if len(value.polynomial) != 2:
        return None
    linear, constant = value.polynomial[0], value.polynomial[1]
    if linear == 0:
        return None
    return Fraction(-constant, linear)


def _rational_prefix(
    value: Fraction, base: int, fractional_places: int
) -> tuple[int, tuple[int, ...]]:
    """Exact prefix under the canonical terminating-zero convention."""

    # The defining interval uses the mathematical floor, not truncation
    # toward zero.  Computing the floor after scaling also handles negative
    # rationals correctly: for example -1/2 in base 10 is -1.500..., not
    # 0.500... (the latter would be a prefix of the wrong number).
    scale = base**fractional_places
    scaled = value * scale
    scaled_floor = scaled.numerator // scaled.denominator
    integer_part, remainder = divmod(scaled_floor, scale)
    fractional_digits: list[int] = []
    for _ in range(fractional_places):
        remainder *= base
        digit, remainder = divmod(remainder, scale)
        fractional_digits.append(int(digit))
    return int(integer_part), tuple(fractional_digits)


def _selected_root_value(value: RealAlgebraicValue) -> Fraction | None:
    """Return a rational root value when the selected root is exactly rational."""

    rational = _rational_value(value)
    if rational is not None:
        return rational
    return None


def _scaled_integer_part(
    value: RealAlgebraicValue,
    *,
    scale: int,
    isolation_bits: int,
) -> int:
    """Exact floor of ``base**places * alpha`` for the selected root.

    If ``alpha`` is a root of ``f`` of degree ``d``, then ``beta = b**q*alpha``
    is a root of ``g(beta) = sum_i c_i b**(q*i) beta**i``, whose coefficients
    are integers.  The selected root index is preserved because ``g`` is a
    positive scalar substitution of ``f``.
    """

    execution = current_request_execution()
    if execution is not None and execution.deadline is not None:
        return run_scaled_integer_part_worker(
            polynomial=value.polynomial,
            real_root_index=value.real_root_index,
            scale=scale,
            isolation_bits=isolation_bits,
            deadline=execution.deadline,
        )
    return _scaled_integer_part_in_process(
        value, scale=scale, isolation_bits=isolation_bits
    )


def _scaled_integer_part_in_process(
    value: RealAlgebraicValue,
    *,
    scale: int,
    isolation_bits: int,
) -> int:
    import sympy

    symbol = sympy.Symbol("x")
    scaled_coefficients = [
        coefficient * scale**position
        for position, coefficient in enumerate(value.polynomial)
    ]
    polynomial = sympy.Poly.from_list(
        [int(coefficient) for coefficient in scaled_coefficients],
        gens=symbol,
        domain=sympy.ZZ,
    )
    intervals = polynomial.intervals()
    if value.real_root_index >= len(intervals):
        raise OperationDomainValidationError(
            location=("value", "real_root_index"),
            code="algebraic_number.radix_root_index",
            message="real_root_index must select an existing real root",
        )
    lower, upper = intervals[value.real_root_index][0]
    max_refinements = max(4_096, (isolation_bits + 7) // 8)
    for _ in range(max_refinements):
        if (candidate := _unique_floor_of_open_interval(lower, upper)) is not None:
            return candidate
        lower, upper = polynomial.refine_root(lower, upper, steps=8)
    raise OperationResourceAdmissionError(
        location=("value",),
        code="algebraic_number.radix_refinement_bound",
        message="root isolation did not separate the scaled value from an integer",
    )


def radix_prefix(
    value: RealAlgebraicValue,
    base: int,
    fractional_places: int,
) -> RadixPrefixResult:
    """Return the exact base-b prefix of one canonical real algebraic value."""

    if not isinstance(value, RealAlgebraicValue):
        raise TypeError("value must be a RealAlgebraicValue")
    if type(base) is not int or type(fractional_places) is not int:
        raise TypeError("base and fractional_places must be integers")
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return radix_prefix(value, base, fractional_places)
    deadline = execution.started_at + 60
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)

    _require_request(base, fractional_places)
    request = RadixPrefixRequest(
        value=value, base=base, fractional_places=fractional_places
    )
    admission = _admit_request(request)
    _admit_real_polynomial(request.value)
    rational = _rational_value(request.value)
    if rational is not None:
        if request.value.real_root_index != 0:
            raise OperationDomainValidationError(
                location=("value", "real_root_index"),
                code="algebraic_number.radix_root_index",
                message="a linear polynomial has exactly one real root",
            )
        integer_part, digits = _rational_prefix(
            rational, request.base, request.fractional_places
        )
        return RadixPrefixResult(
            value=request.value,
            base=request.base,
            fractional_places=request.fractional_places,
            integer_part=integer_part,
            fractional_digits=digits,
        )
    scaled = _scaled_integer_part(
        request.value,
        scale=admission.scale,
        isolation_bits=admission.isolation_bits,
    )
    scale = admission.scale
    integer_part, remainder = divmod(scaled, scale)
    fractional_digits: list[int] = []
    for _ in range(request.fractional_places):
        remainder *= request.base
        digit, remainder = divmod(remainder, scale)
        fractional_digits.append(int(digit))
    return RadixPrefixResult(
        value=request.value,
        base=request.base,
        fractional_places=request.fractional_places,
        integer_part=integer_part,
        fractional_digits=tuple(fractional_digits),
    )
