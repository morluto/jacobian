"""Exact bounded enclosure of Kempner digit-family reciprocal series."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

from jacobian._exact import CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory._kempner_models import (
    MAX_KEMPNER_BASE,
    KempnerDigitSet,
)
from jacobian.math.number_theory.kempner._models import (
    MAX_KEMPNER_SERIES_DIGITS,
    MAX_KEMPNER_SERIES_NUMERALS,
    KempnerSeriesEnclosure,
)


def _require_canonical_digit_set(digit_set: KempnerDigitSet) -> None:
    """Reject a non-canonical or out-of-domain digit set at the boundary."""

    if (
        not isinstance(digit_set, KempnerDigitSet)
        or not isinstance(digit_set.base, int)
        or isinstance(digit_set.base, bool)
        or not 2 <= digit_set.base <= MAX_KEMPNER_BASE
        or not isinstance(digit_set.allowed_digits, tuple)
        or not digit_set.allowed_digits
        or any(
            not isinstance(digit, int)
            or isinstance(digit, bool)
            or digit < 0
            or digit >= digit_set.base
            for digit in digit_set.allowed_digits
        )
    ):
        raise OperationDomainValidationError(
            location=("digit_set",),
            code="number_theory.kempner_series.canonical_digit_set",
            message="digit_set must be a canonical proper digit subset",
        )
    if (
        digit_set.allowed_digits != tuple(sorted(set(digit_set.allowed_digits)))
        or len(digit_set.allowed_digits) >= digit_set.base
    ):
        raise OperationDomainValidationError(
            location=("digit_set",),
            code="number_theory.kempner_series.canonical_digit_set",
            message="digit_set must be a canonical proper digit subset",
        )


def _numeral_count(digit_count: int, nonzero_count: int, cutoff: int) -> int:
    """Return ``r * sum_{m<cutoff} s^m``, the finite family size through D."""

    if nonzero_count == 0 or cutoff == 0:
        return 0
    if digit_count == 1:
        return nonzero_count * cutoff
    geometric = (pow(digit_count, cutoff) - 1) // (digit_count - 1)
    return int(nonzero_count * geometric)


def require_series_admission(digit_set: KempnerDigitSet, cutoff: int) -> int:
    """Preflight numeral count, rational height, and wire bytes once per call.

    Returns the admitted finite numeral count. Every ``m``-digit accepted
    numeral is below ``b^m``, so each reciprocal contributes at most
    ``D * digits(b)`` denominator digits and the reduced partial sum keeps
    both components at or below ``MAX_KEMPNER_SERIES_DIGITS`` digits
    whenever ``count * D * digits(b) + digits(count)`` does. The tail is a
    fixed short rational in ``b``, ``s``, ``r``, and ``D``. The canonical
    wire encoding of the four rationals plus the source digit set is linear
    in those same quantities.
    """

    _require_canonical_digit_set(digit_set)
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 0:
        raise OperationDomainValidationError(
            location=("cutoff",),
            code="number_theory.kempner_series.cutoff",
            message="cutoff must be a nonnegative integer",
        )
    base = digit_set.base
    digit_count = len(digit_set.allowed_digits)
    nonzero_count = sum(1 for digit in digit_set.allowed_digits if digit != 0)
    count = _numeral_count(digit_count, nonzero_count, cutoff)
    if count > MAX_KEMPNER_SERIES_NUMERALS:
        raise OperationResourceAdmissionError(
            location=("cutoff",),
            code="number_theory.kempner_series.numeral_count",
            message=(
                "the finite numeral family through the cutoff exceeds the "
                f"{MAX_KEMPNER_SERIES_NUMERALS}-term enumeration envelope"
            ),
        )
    height_digits = (
        count * max(cutoff, 1) * len(str(base)) + len(str(max(count, 1)))
        if count
        else 1
    )
    if height_digits > MAX_KEMPNER_SERIES_DIGITS:
        raise OperationResourceAdmissionError(
            location=("digit_set", "cutoff"),
            code="number_theory.kempner_series.rational_height",
            message=(
                "the exact-rational height through the cutoff exceeds the "
                f"{MAX_KEMPNER_SERIES_DIGITS}-digit result envelope"
            ),
        )
    return count


def _tail_bound(
    base: int, digit_count: int, nonzero_count: int, cutoff: int
) -> Fraction:
    """Return the exact rational ``r*(s/b)^D/(1-s/b)`` tail bound.

    Every ``m``-digit member is at least ``b^(m-1)`` and there are exactly
    ``r*s^(m-1)`` of them, so the omitted reciprocal mass past ``D`` digits
    is at most ``sum_{m>D} r*s^(m-1)/b^(m-1)``, the stated geometric sum.
    The proper-subset condition (``s < b``) keeps the ratio finite.
    """

    if nonzero_count == 0:
        return Fraction(0)
    return Fraction(
        nonzero_count * digit_count**cutoff * base,
        base**cutoff * (base - digit_count),
    )


def enclose_kempner_series(
    digit_set: KempnerDigitSet, cutoff: int
) -> KempnerSeriesEnclosure:
    """Enclose the reciprocal series of one Kempner family through D digits."""

    count = require_series_admission(digit_set, cutoff)
    base = digit_set.base
    allowed = digit_set.allowed_digits
    nonzero = tuple(digit for digit in allowed if digit != 0)
    partial = Fraction(0)
    enumerated = 0
    for length in range(1, cutoff + 1):
        for first in nonzero:
            for rest in product(allowed, repeat=length - 1):
                value = first
                for digit in rest:
                    value = value * base + digit
                partial += Fraction(1, value)
                enumerated += 1
                if enumerated % 4_096 == 0:
                    request_checkpoint("during Kempner series enumeration")
    if enumerated != count:
        raise RuntimeError("Kempner enumeration missed its admitted numeral count")
    tail = _tail_bound(base, len(allowed), len(nonzero), cutoff)
    upper = partial + tail
    return KempnerSeriesEnclosure._from_kernel(
        digit_set,
        cutoff,
        partial_sum=CanonicalRational.from_fraction(partial),
        tail_upper_bound=CanonicalRational.from_fraction(tail),
        lower=CanonicalRational.from_fraction(partial),
        upper=CanonicalRational.from_fraction(upper),
    )


__all__ = [
    "enclose_kempner_series",
    "require_series_admission",
]
