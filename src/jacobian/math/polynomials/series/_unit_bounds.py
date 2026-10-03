"""Whole-kernel envelopes for multiplicative inverse and quotient series."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from re import fullmatch
from typing import TYPE_CHECKING, Literal

from jacobian._exact import CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)

if TYPE_CHECKING:
    from ._models import TruncatedSeries

MAX_UNIT_ORDER = 2048
MAX_UNIT_WORK = 1 << 28
MAX_UNIT_LIMB_WORK = 1 << 36
MAX_UNIT_SCRATCH_BITS = 1 << 18
MAX_UNIT_STORAGE_BITS = 1 << 32


def _reject(
    reason: str, message: str, *, family: Literal["unit", "power"] = "unit"
) -> None:
    raise OperationResourceAdmissionError(
        location=(), code=f"formal_power_series.{family}_{reason}", message=message
    )


def _invalid() -> None:
    raise OperationDomainValidationError(
        location=(),
        code="formal_power_series.source_structure",
        message="series need canonical scalar coefficients and a complete ordered prefix",
    )


def require_series(
    series: TruncatedSeries,
    *,
    maximum_digits: int,
    resource_family: Literal["unit", "power"] = "unit",
) -> None:
    """Bound native copies before shape shortcuts, LCMs, and backend conversion."""
    from ._models import MAX_TRUNCATE_SOURCE_ORDER, TruncatedSeries

    if (
        type(series) is not TruncatedSeries
        or type(getattr(series, "truncation_order", None)) is not int
        or series.truncation_order < 1
        or type(getattr(series, "coefficients", None)) is not tuple
        or type(getattr(series, "variable", None)) is not str
        or len(series.variable) > 32
        or fullmatch(r"[A-Za-z][A-Za-z0-9_]*", series.variable) is None
        or len(series.coefficients) != series.truncation_order
    ):
        _invalid()
    if series.truncation_order > MAX_TRUNCATE_SOURCE_ORDER:
        _reject(
            "order",
            "source exceeds the bounded series carrier admission",
            family=resource_family,
        )
    ceiling = 10**maximum_digits
    for value in series.coefficients:
        request_checkpoint("during exact unit-series scalar admission")
        if (
            type(value) is not CanonicalRational
            or type(getattr(value, "num", None)) is not int
            or type(getattr(value, "den", None)) is not int
            or value.den <= 0
        ):
            _invalid()
        if (
            value.num.bit_length() > ceiling.bit_length()
            or value.den.bit_length() > ceiling.bit_length()
            or abs(value.num) >= ceiling
            or value.den >= ceiling
        ):
            _reject(
                "coefficient",
                f"coefficient exceeds {maximum_digits} digits",
                family=resource_family,
            )
        if gcd(value.num, value.den) != 1:
            _invalid()


@dataclass(frozen=True)
class Envelope:
    """Coefficient-level work and exact integer payload, not bit time or RSS."""

    work: int
    bits: int
    storage: int

    def checked(self) -> None:
        for name, amount, limit in (
            ("work", self.work, MAX_UNIT_WORK),
            ("scratch", self.bits, MAX_UNIT_SCRATCH_BITS),
            ("limb_work", self.work * ((self.bits + 63) // 64), MAX_UNIT_LIMB_WORK),
            ("storage", self.storage, MAX_UNIT_STORAGE_BITS),
        ):
            if amount > limit:
                _reject(name, f"proved unit-series {name} {amount} exceeds {limit}")


def _norm(series: TruncatedSeries) -> tuple[int, int, int]:
    """Common denominator, numerator l1 norm, and constant magnitude, in bits.

    Check a prospective LCM before constructing it. The numerator sum is
    bounded before expansion too; all these scalars are private admission work.
    """
    denominator = 1
    for value in series.coefficients:
        request_checkpoint("during unit-series denominator admission")
        quotient = denominator // gcd(denominator, value.den)
        if quotient.bit_length() + value.den.bit_length() > MAX_UNIT_SCRATCH_BITS:
            _reject("scratch", "common-denominator clearing exceeds scratch bound")
        denominator = quotient * value.den
    bits = max(abs(value.num).bit_length() for value in series.coefficients)
    if (
        bits + denominator.bit_length() + len(series.coefficients).bit_length()
        > MAX_UNIT_SCRATCH_BITS
    ):
        _reject("scratch", "cleared numerator norm exceeds scratch bound")
    norm = sum(
        abs(value.num) * (denominator // value.den) for value in series.coefficients
    )
    constant = abs(series.coefficients[0].num) * (
        denominator // series.coefficients[0].den
    )
    return max(1, norm.bit_length()), denominator.bit_length(), constant.bit_length()


def _product(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return left[0] + right[0], left[1] + right[1]


def _difference(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return max(left[0] + right[1], right[0] + left[1]) + 1, left[1] + right[1]


def kernel_envelope(
    denominator: TruncatedSeries, numerator: TruncatedSeries | None = None
) -> Envelope:
    """Bound Newton's full products, quotient, replay, and retained values.

    For B=P/D, p=|P[0]| and C=||P||_1, every inverse prefix has a
    numerator l1 norm <= N*D*C**(N-1) over p**N. At each doubling the
    prefix is already correct. Thus X*(2-B*X), including discarded high
    coefficients, is bounded by ordinary l1 product/subtraction rules.
    Quotient and residual use those same rules without relying on cancellation.

    Two products per doubling have geometrically increasing supports; their
    classical coefficient-operation sum, conversion, final product/replay,
    normalization and copies fit 32*N**2. A separate limb-width-weighted
    proxy charges that work times ceil(scratch_bits/64); neither accounting
    unit claims an execution time or exact CPU bit-operation count.
    At most 32*N coefficient slots
    are simultaneously live across the full products and retained values.
    Constant denominators use only independent scalar arithmetic and O(N) work.
    """
    n = denominator.truncation_order
    if not any(value.num for value in denominator.coefficients[1:]):
        values = denominator.coefficients + (
            () if numerator is None else numerator.coefficients
        )
        bits = (
            4 * max(max(abs(v.num).bit_length(), v.den.bit_length()) for v in values)
            + 4
        )
        return Envelope(32 * n, bits, 32 * n * bits)
    norm, den, constant = _norm(denominator)
    inverse = (n.bit_length() + den + (n - 1) * norm, n * constant)
    source = (norm, den)
    product = _product(source, inverse)
    correction = _difference((2, 1), product)
    newton = _product(inverse, correction)
    if numerator is None:
        quotient = inverse
        original = (1, 1)
    else:
        a, b, _ = _norm(numerator)
        original = (a, b)
        quotient = _product(original, inverse)
    residual = _difference(_product(source, quotient), original)
    bits = max(*newton, *quotient, *residual, *source, *original)
    return Envelope(32 * n * n, bits, 32 * n * bits)


def admit_replay(
    denominator: TruncatedSeries,
    result: TruncatedSeries,
    numerator: TruncatedSeries | None = None,
) -> None:
    """A supplied result has its own bounds; the producer proof cannot price it."""
    require_series(result, maximum_digits=4096)
    if (
        result.variable != denominator.variable
        or result.truncation_order != denominator.truncation_order
    ):
        _invalid()
    n = denominator.truncation_order
    if not any(value.num for value in denominator.coefficients[1:]):
        scalar = denominator.coefficients[0]
        source = (abs(scalar.num).bit_length(), scalar.den.bit_length())
        claimed = (
            max(abs(value.num).bit_length() for value in result.coefficients),
            max(value.den.bit_length() for value in result.coefficients),
        )
        original = (
            (1, 1)
            if numerator is None
            else (
                max(abs(value.num).bit_length() for value in numerator.coefficients),
                max(value.den.bit_length() for value in numerator.coefficients),
            )
        )
        bits = max(
            *_difference(_product(source, claimed), original),
            *source,
            *claimed,
            *original,
        )
        Envelope(32 * n, bits, 32 * n * bits).checked()
        return
    b, d, _ = _norm(denominator)
    q, e, _ = _norm(result)
    original = (1, 1) if numerator is None else _norm(numerator)[:2]
    residual = _difference(_product((b, d), (q, e)), original)
    bits = max(*residual, b, d, q, e, *original)
    Envelope(32 * n * n, bits, 32 * n * bits).checked()
