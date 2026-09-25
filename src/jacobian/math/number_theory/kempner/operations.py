"""Exact bounded enclosure of Kempner digit-family reciprocal series."""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.analysis.intervals import ClosedRationalInterval
from jacobian.math.number_theory._kempner_models import (
    MAX_KEMPNER_BASE,
    KempnerDigitSet,
)
from jacobian.math.number_theory.kempner._models import (
    MAX_KEMPNER_DECIMAL_DIGITS,
    MAX_KEMPNER_DECIMAL_STACK,
    MAX_KEMPNER_DECIMAL_TERMS,
    MAX_KEMPNER_DECIMAL_VISITS,
    MAX_KEMPNER_SERIES_DIGITS,
    MAX_KEMPNER_SERIES_NUMERALS,
    KempnerDecimalEnclosure,
    KempnerSeriesEnclosure,
)

# The admission preflight may factor every accepted numeral while sieving
# through this range. Both the numeral count and maximum numeral are bounded
# before the preflight starts.
_MAX_LCM_SIEVE_BOUND = 2 * MAX_KEMPNER_SERIES_NUMERALS


def _record_prefix_prime_powers(
    prefix: int,
    remaining: int,
    digit_set: KempnerDigitSet,
    smallest_factor: list[int],
    prime_powers: dict[int, int],
) -> int:
    """Record denominator prime powers below one canonical prefix."""

    if remaining:
        return sum(
            _record_prefix_prime_powers(
                prefix * digit_set.base + digit,
                remaining - 1,
                digit_set,
                smallest_factor,
                prime_powers,
            )
            for digit in digit_set.allowed_digits
        )
    value = prefix
    while value > 1:
        prime = smallest_factor[value]
        power = 1
        while value % prime == 0:
            value //= prime
            power *= prime
        if power > prime_powers.get(prime, 1):
            prime_powers[prime] = power
    return 1


def _family_lcm_digit_bound(
    digit_set: KempnerDigitSet, cutoff: int, maximum: int, expected_count: int
) -> int:
    """Bound decimal digits of the lcm of all accepted numerals through D.

    A smallest-prime-factor sieve factors each admitted numeral. The maximum
    prime power seen for every prime determines the exact lcm; summing their
    bit lengths bounds its decimal digits without constructing the lcm.
    """

    if maximum < 2:
        return 1
    request_checkpoint("during Kempner denominator sieve")
    smallest_factor = list(range(maximum + 1))
    for prime in range(2, isqrt(maximum) + 1):
        if smallest_factor[prime] == prime:
            for multiple in range(prime * prime, maximum + 1, prime):
                if smallest_factor[multiple] == multiple:
                    smallest_factor[multiple] = prime
    request_checkpoint("during Kempner denominator sieve")

    prime_powers: dict[int, int] = {}
    enumerated = 0
    checkpoint_at = 4_096

    nonzero = tuple(digit for digit in digit_set.allowed_digits if digit)
    for length in range(1, cutoff + 1):
        for first in nonzero:
            enumerated += _record_prefix_prime_powers(
                first, length - 1, digit_set, smallest_factor, prime_powers
            )
            if enumerated >= checkpoint_at:
                request_checkpoint("during Kempner denominator admission")
                checkpoint_at = enumerated + 4_096
    if enumerated != expected_count:
        raise RuntimeError("Kempner denominator admission missed an accepted numeral")

    # The lcm is the product of the maximum prime power seen for each prime.
    # Summing bit lengths upper-bounds the product's bit length, and
    # 30103/100000 is a strict upper bound for log10(2).
    bit_bound = sum(power.bit_length() for power in prime_powers.values())
    return (bit_bound * 30_103 + 99_999) // 100_000


def _require_canonical_digit_set(digit_set: KempnerDigitSet) -> None:
    """Reject a non-canonical or out-of-domain digit set at the boundary."""

    base = getattr(digit_set, "base", None)
    allowed_digits = getattr(digit_set, "allowed_digits", None)
    if (
        not isinstance(digit_set, KempnerDigitSet)
        or type(base) is not int
        or not 2 <= base <= MAX_KEMPNER_BASE
        or type(allowed_digits) is not tuple
        or not allowed_digits
        or any(
            type(digit) is not int or digit < 0 or digit >= base
            for digit in allowed_digits
        )
    ):
        raise OperationDomainValidationError(
            location=("digit_set",),
            code="number_theory.kempner_series.canonical_digit_set",
            message="digit_set must be a canonical proper digit subset",
        )
    if (
        allowed_digits != tuple(sorted(set(allowed_digits)))
        or len(allowed_digits) >= base
    ):
        raise OperationDomainValidationError(
            location=("digit_set",),
            code="number_theory.kempner_series.canonical_digit_set",
            message="digit_set must be a canonical proper digit subset",
        )


def _numeral_count(digit_count: int, nonzero_count: int, cutoff: int) -> int:
    """Count admitted numerals, saturating above the published work limit."""

    if nonzero_count == 0 or cutoff == 0:
        return 0
    if digit_count == 1:
        return min(
            nonzero_count * cutoff,
            MAX_KEMPNER_SERIES_NUMERALS + 1,
        )
    count = 0
    layer_count = nonzero_count
    for _ in range(cutoff):
        count += layer_count
        if count > MAX_KEMPNER_SERIES_NUMERALS:
            return MAX_KEMPNER_SERIES_NUMERALS + 1
        layer_count *= digit_count
    return count


def require_series_admission(digit_set: KempnerDigitSet, cutoff: int) -> int:
    """Preflight numeral count, rational height, and wire bytes once per call.

    Returns the admitted finite numeral count. The partial-sum denominator
    divides the lcm of the accepted numerals; a bounded smallest-prime-factor
    sieve derives that exact lcm's prime powers before enumeration. The sum's
    numerator is at most ``count`` times that denominator. The geometric tail
    has a denominator controlled by ``b``, ``s``, ``r``, and ``D``; the
    resulting endpoint heights are preflighted with the canonical wire size.
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
    # The partial denominator divides the lcm of the accepted numerals. A
    # smallest-prime-factor sieve over the bounded family gives a substantially
    # tighter exact-height estimate than lcm(1, ..., base**cutoff - 1). The
    # tail is then added with its own denominator; cap the power before
    # constructing an enormous maximum.
    if count:
        tail_digits = cutoff * len(str(base)) + len(str(base)) + 1
        power_cap = max(
            (MAX_KEMPNER_SERIES_DIGITS * 8) // 5 + 16,
            _MAX_LCM_SIEVE_BOUND + 1,
        )
        bounded_power = 1
        factor = base
        exponent = cutoff
        while exponent:
            if exponent & 1:
                bounded_power *= factor
                if bounded_power > power_cap:
                    break
            exponent >>= 1
            if exponent:
                factor *= factor
                if factor > power_cap:
                    factor = power_cap + 1
        if bounded_power > power_cap:
            raise OperationResourceAdmissionError(
                location=("digit_set", "cutoff"),
                code="number_theory.kempner_series.rational_height",
                message=(
                    "the exact-rational height through the cutoff exceeds the "
                    f"{MAX_KEMPNER_SERIES_DIGITS}-digit result envelope"
                ),
            )
        max_n = bounded_power - 1
        lcm_digits = _family_lcm_digit_bound(digit_set, cutoff, max_n, count)
        height_digits = lcm_digits + tail_digits + len(str(count)) + 2
    else:
        height_digits = 1
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
    # Prefix recurrence: each accepted numeral is reached exactly once by
    # extending an admitted prefix.  No dense product/list is materialised,
    # which keeps the formerly rejected dense family within the bounded
    # request envelope while preserving the exact partial sum.
    partial = Fraction(0)
    enumerated = 0

    def extend(prefix: int, remaining: int) -> None:
        nonlocal partial, enumerated
        if remaining == 0:
            partial += Fraction(1, prefix)
            enumerated += 1
            if enumerated % 4_096 == 0:
                request_checkpoint("during Kempner prefix recurrence")
            return
        for digit in allowed:
            extend(prefix * base + digit, remaining - 1)

    for length in range(1, cutoff + 1):
        for first in nonzero:
            extend(first, length - 1)
    if enumerated != count:
        raise RuntimeError("Kempner enumeration missed its admitted numeral count")
    tail = _tail_bound(base, len(allowed), len(nonzero), cutoff)
    upper = partial + tail

    def canonical(value: Fraction, label: str) -> CanonicalRational:
        try:
            return CanonicalRational.from_fraction(value)
        except (ValidationError, ValueError, OverflowError) as error:
            raise OperationResourceAdmissionError(
                location=(label,),
                code="number_theory.kempner_series.rational_height",
                message=(
                    "the exact-rational result exceeds the "
                    f"{MAX_KEMPNER_SERIES_DIGITS}-digit result envelope"
                ),
            ) from error

    return KempnerSeriesEnclosure._from_kernel(
        digit_set,
        cutoff,
        partial_sum=canonical(partial, "partial_sum"),
        tail_upper_bound=canonical(tail, "tail_upper_bound"),
        lower=canonical(partial, "lower"),
        upper=canonical(upper, "upper"),
    )


def _decimal_numeral_count(digit_count: int, nonzero_count: int, cutoff: int) -> int:
    """Count the family through D digits, saturating beyond its term limit."""

    count = 0
    layer_count = nonzero_count
    for _ in range(cutoff):
        count += layer_count
        if count > MAX_KEMPNER_DECIMAL_TERMS:
            return count
        layer_count *= digit_count
    return count


def _decimal_node_visit_bound(digit_count: int, nonzero_count: int, cutoff: int) -> int:
    """Count every prefix node visited by the per-length depth-first walk."""

    layer_count = nonzero_count
    nodes_per_length = 0
    node_visits = 0
    for _ in range(cutoff):
        nodes_per_length += layer_count
        node_visits += nodes_per_length
        if node_visits > MAX_KEMPNER_DECIMAL_VISITS:
            return node_visits
        layer_count *= digit_count
    return node_visits


def _admit_decimal_series(
    digit_set: KempnerDigitSet, cutoff: int, precision: int
) -> tuple[int, tuple[int, ...]]:
    """Admit scalar inputs, total work, traversal storage, and rational height."""

    _require_canonical_digit_set(digit_set)
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, int)
        or not 0 <= cutoff <= 999
    ):
        raise OperationDomainValidationError(
            location=("cutoff",),
            code="number_theory.kempner_series.cutoff",
            message="cutoff must be an integer from 0 through 999",
        )
    if (
        isinstance(precision, bool)
        or not isinstance(precision, int)
        or not (1 <= precision <= MAX_KEMPNER_DECIMAL_DIGITS)
    ):
        raise OperationDomainValidationError(
            location=("precision",),
            code="number_theory.kempner_series.decimal_precision",
            message=(
                "precision must be an integer between 1 and "
                f"{MAX_KEMPNER_DECIMAL_DIGITS}"
            ),
        )
    base = digit_set.base
    allowed = digit_set.allowed_digits
    nonzero = tuple(digit for digit in allowed if digit)
    count = _decimal_numeral_count(len(allowed), len(nonzero), cutoff)
    if count > MAX_KEMPNER_DECIMAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("cutoff",),
            code="number_theory.kempner_series.decimal_work",
            message=(
                "the finite numeral family exceeds the "
                f"{MAX_KEMPNER_DECIMAL_TERMS}-term fixed-point envelope"
            ),
        )
    node_visits = _decimal_node_visit_bound(len(allowed), len(nonzero), cutoff)
    if node_visits > MAX_KEMPNER_DECIMAL_VISITS:
        raise OperationResourceAdmissionError(
            location=("cutoff", "digit_set"),
            code="number_theory.kempner_series.decimal_work",
            message=(
                "the prefix recurrence exceeds the "
                f"{MAX_KEMPNER_DECIMAL_VISITS}-node work envelope"
            ),
        )
    stack_bound = cutoff * max(len(allowed) - 1, 0) + 1
    if stack_bound > MAX_KEMPNER_DECIMAL_STACK:
        raise OperationResourceAdmissionError(
            location=("cutoff", "digit_set"),
            code="number_theory.kempner_series.decimal_stack",
            message=(
                "the prefix traversal stack exceeds the "
                f"{MAX_KEMPNER_DECIMAL_STACK}-entry envelope"
            ),
        )
    # The exact tail denominator is base**cutoff * (base - digit_count).
    # A conservative sum-of-heights bound admits it before exponentiation.
    tail_digits = cutoff * len(str(base)) + len(str(base - len(allowed)))
    estimated_digits = precision + tail_digits + len(str(max(count, 1))) + 8
    if estimated_digits > MAX_KEMPNER_SERIES_DIGITS:
        raise OperationResourceAdmissionError(
            location=("cutoff", "precision"),
            code="number_theory.kempner_series.decimal_height",
            message=(
                "the exact rational interval exceeds the "
                f"{MAX_KEMPNER_SERIES_DIGITS}-digit result envelope"
            ),
        )
    return count, nonzero


def enclose_kempner_series_decimal(
    digit_set: KempnerDigitSet, cutoff: int, precision: int
) -> KempnerDecimalEnclosure:
    """Return an exact interval using fixed-point bounds for each reciprocal.

    Unlike the exact-partial-sum operation, this does not construct the huge
    common denominator of a dense family. At scale 10**precision each term
    1/n contributes floor(scale/n) below and ceil(scale/n) above; summing
    those inequalities and adding the exact geometric tail gives the result.
    """

    count, nonzero = _admit_decimal_series(digit_set, cutoff, precision)
    if not nonzero:
        zero = CanonicalRational(num=0, den=1)
        return KempnerDecimalEnclosure(
            digit_set=digit_set,
            cutoff=cutoff,
            precision=precision,
            enclosure=ClosedRationalInterval(lower=zero, upper=zero),
        )
    base = digit_set.base
    allowed = digit_set.allowed_digits

    scale = 10**precision
    lower_units = 0
    upper_units = 0
    enumerated = 0
    node_visits = 0

    for length in range(1, cutoff + 1):
        for first in nonzero:
            stack = [(first, length - 1)]
            while stack:
                prefix, remaining = stack.pop()
                node_visits += 1
                if node_visits % 4_096 == 0:
                    request_checkpoint("during Kempner fixed-point traversal")
                if remaining == 0:
                    quotient, remainder = divmod(scale, prefix)
                    lower_units += quotient
                    upper_units += quotient + bool(remainder)
                    enumerated += 1
                    continue
                stack.extend(
                    (prefix * base + digit, remaining - 1)
                    for digit in reversed(allowed)
                )
    if enumerated != count:
        raise RuntimeError("Kempner fixed-point recurrence missed its admitted count")

    tail = _tail_bound(base, len(allowed), len(nonzero), cutoff)
    lower = Fraction(lower_units, scale)
    upper = Fraction(upper_units, scale) + tail
    return KempnerDecimalEnclosure(
        digit_set=digit_set,
        cutoff=cutoff,
        precision=precision,
        enclosure=ClosedRationalInterval(
            lower=CanonicalRational.from_fraction(lower),
            upper=CanonicalRational.from_fraction(upper),
        ),
    )


__all__ = [
    "enclose_kempner_series",
    "enclose_kempner_series_decimal",
    "require_series_admission",
]
