"""Independent exact checks for claimed LOG and SQRT point enclosures.

For ``1 <= y < 2``, set ``z = (y - 1) / (y + 1)``.  DLMF 4.37.E24
and 4.38.E5 give

``log(y) = 2 * atanh(z) = 2 * sum(z**(2*j + 1) / (2*j + 1))``.

Here ``0 <= z < 1/3``.  After ``n`` terms, replacing every omitted
denominator by ``2*n + 1`` and summing the remaining geometric powers gives
the exact rational tail bound used below::

    remainder <= 2*z**(2*n + 1) / ((2*n + 1)*(1 - z**2)).

The identities are documented at https://dlmf.nist.gov/4.37.E24 and
https://dlmf.nist.gov/4.38.E5.  The checker uses only ``Fraction`` arithmetic;
it does not call the Arb producer that may have supplied the claimed interval.
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

from jacobian.math.analysis._point_enclosure import (
    MAX_POINT_CHECK_LOG_TERMS,
    PointEnclosureCheckOutcome,
    PointEnclosureCheckRequest,
    RealUnaryFunction,
)


def _positive_atanh_enclosures(
    z: Fraction,
) -> Iterator[tuple[Fraction, Fraction]]:
    """Yield exact lower and upper bounds after each admitted series term."""

    if not 0 <= z <= Fraction(1, 3):
        raise ValueError("positive atanh enclosure requires 0 <= z <= 1/3")

    z_squared = z * z
    next_power = z
    partial_sum = Fraction(0)
    for index in range(MAX_POINT_CHECK_LOG_TERMS):
        partial_sum += next_power / (2 * index + 1)
        next_power *= z_squared
        first_omitted_denominator = 2 * (index + 1) + 1
        lower = 2 * partial_sum
        tail = 2 * next_power / first_omitted_denominator / (1 - z_squared)
        yield lower, lower + tail


def _power_of_two(exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(1 << exponent)
    return Fraction(1, 1 << -exponent)


def _log_range_reduction(argument: Fraction) -> tuple[int, Fraction]:
    """Return the unique ``k, y`` with ``argument = 2**k*y`` and 1 <= y < 2."""

    numerator_bits = argument.numerator.bit_length()
    denominator_bits = argument.denominator.bit_length()
    exponent = numerator_bits - denominator_bits
    scale = _power_of_two(exponent)
    if argument < scale:
        exponent -= 1
        scale /= 2
    reduced = argument / scale
    if not 1 <= reduced < 2:
        raise ArithmeticError("exact power-of-two range reduction failed")
    return exponent, reduced


def _claim_relation(
    claimed_lower: Fraction,
    claimed_upper: Fraction,
    proven_lower: Fraction,
    proven_upper: Fraction,
) -> PointEnclosureCheckOutcome | None:
    if claimed_lower <= proven_lower and proven_upper <= claimed_upper:
        return "ACCEPTED"
    if claimed_upper < proven_lower or proven_upper < claimed_lower:
        return "REJECTED"
    return None


def _sqrt_outcome(
    argument: Fraction,
    claimed_lower: Fraction,
    claimed_upper: Fraction,
) -> PointEnclosureCheckOutcome:
    if argument < 0:
        return "REJECTED"

    lower_holds = claimed_lower <= 0 or claimed_lower * claimed_lower <= argument
    upper_holds = claimed_upper >= 0 and claimed_upper * claimed_upper >= argument
    return "ACCEPTED" if lower_holds and upper_holds else "REJECTED"


def _log_outcome(
    argument: Fraction,
    claimed_lower: Fraction,
    claimed_upper: Fraction,
) -> PointEnclosureCheckOutcome:
    if argument <= 0:
        return "REJECTED"

    exponent, reduced = _log_range_reduction(argument)
    reduced_z = (reduced - 1) / (reduced + 1)
    reduced_enclosures = _positive_atanh_enclosures(reduced_z)
    if exponent == 0:
        for proven_lower, proven_upper in reduced_enclosures:
            relation = _claim_relation(
                claimed_lower, claimed_upper, proven_lower, proven_upper
            )
            if relation is not None:
                return relation
        return "NON_RESULT"

    log_two_enclosures = _positive_atanh_enclosures(Fraction(1, 3))
    for (reduced_lower, reduced_upper), (
        log_two_lower,
        log_two_upper,
    ) in zip(reduced_enclosures, log_two_enclosures, strict=True):
        if exponent > 0:
            proven_lower = reduced_lower + exponent * log_two_lower
            proven_upper = reduced_upper + exponent * log_two_upper
        else:
            proven_lower = reduced_lower + exponent * log_two_upper
            proven_upper = reduced_upper + exponent * log_two_lower
        relation = _claim_relation(
            claimed_lower, claimed_upper, proven_lower, proven_upper
        )
        if relation is not None:
            return relation
    return "NON_RESULT"


def point_enclosure_check_outcome(
    request: PointEnclosureCheckRequest,
) -> PointEnclosureCheckOutcome:
    """Replay one structurally admitted enclosure claim deterministically."""

    enclosure = request.enclosure
    if enclosure.lower.compare(enclosure.upper) > 0:
        return "REJECTED"

    argument = enclosure.argument.as_fraction()
    claimed_lower = enclosure.lower.as_fraction()
    claimed_upper = enclosure.upper.as_fraction()
    if enclosure.function is RealUnaryFunction.SQRT:
        return _sqrt_outcome(argument, claimed_lower, claimed_upper)
    return _log_outcome(argument, claimed_lower, claimed_upper)


__all__ = ["point_enclosure_check_outcome"]
