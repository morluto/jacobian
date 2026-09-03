"""Native arithmetic-function operations over canonical rational values."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import gcd

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math._rational_height import RationalHeight, sum_heights
from jacobian.math.number_theory.arithmetic_functions._models import (
    _MAX_DIVISOR_PREFIX_LENGTH,
    _MAX_SUMMATORY_LENGTH,
)

MAX_DIVISOR_INCIDENCES = 600_000


@dataclass(frozen=True)
class _ConvolutionPlan:
    incidences: tuple[tuple[int, int], ...]
    left_nums: tuple[int, ...]
    left_dens: tuple[int, ...]
    right_nums: tuple[int, ...]
    right_dens: tuple[int, ...]
    slot_lcms: tuple[int, ...]


@dataclass(frozen=True)
class _MobiusPlan:
    incidences: tuple[tuple[int, int], ...]
    numerators: tuple[int, ...]
    denominators: tuple[int, ...]
    shared_lcm: int | None


def _rational(value: Fraction | int) -> CanonicalRational:
    """Convert an exact Fraction or int to a CanonicalRational."""
    return CanonicalRational.from_fraction(Fraction(value))


def _divisor_incidences(
    length: int, *, minimum_divisor: int = 1
) -> tuple[tuple[int, int], ...]:
    """Return ``(divisor, multiple)`` pairs for the bounded dense prefix."""
    return tuple(
        (divisor, multiple)
        for divisor in range(minimum_divisor, length + 1)
        for multiple in range(divisor, length + 1, divisor)
    )


def _divisor_incidence_count(length: int, *, minimum_divisor: int = 1) -> int:
    return sum(length // divisor for divisor in range(minimum_divisor, length + 1))


def _require_divisor_incidences(
    length: int, *, minimum_divisor: int = 1
) -> tuple[tuple[int, int], ...]:
    if _divisor_incidence_count(length, minimum_divisor=minimum_divisor) > (
        MAX_DIVISOR_INCIDENCES
    ):
        raise OperationDomainValidationError(
            location=("values",),
            code="arithmetic_functions.divisor_incidence_work_exceeded",
            message=(
                "arithmetic-function prefix exceeds the "
                f"{MAX_DIVISOR_INCIDENCES}-incidence divisor-sieve budget"
            ),
        )
    return _divisor_incidences(length, minimum_divisor=minimum_divisor)


class _HeightSums:
    """Incremental form of ``sum_heights`` for one dense result vector."""

    def __init__(
        self,
        length: int,
        *,
        shared_lcm: int | None = None,
        slot_lcms: tuple[int | None, ...] | None = None,
    ) -> None:
        self.denominator_digits = [0] * length
        self.maximum_adjusted_numerator = [0] * length
        self.maximum_numerator = [0] * length
        self.term_counts = [0] * length
        self.shared_lcm = shared_lcm
        self.slot_lcms = slot_lcms
        self._shared_lift_digits: dict[int, int] = {}

    def add(
        self,
        index: int,
        height: RationalHeight,
        *,
        denominator: int | None = None,
    ) -> None:
        self._add_digits(
            index,
            numerator_digits=height.numerator_digits,
            denominator_digits=height.denominator_digits,
            denominator=denominator,
        )

    def add_product(
        self,
        index: int,
        left: RationalHeight,
        right: RationalHeight,
        *,
        denominator: int | None = None,
    ) -> None:
        self._add_digits(
            index,
            numerator_digits=left.numerator_digits + right.numerator_digits,
            denominator_digits=(left.denominator_digits + right.denominator_digits),
            denominator=denominator,
        )

    def _add_digits(
        self,
        index: int,
        *,
        numerator_digits: int,
        denominator_digits: int,
        denominator: int | None,
    ) -> None:
        self.denominator_digits[index] += denominator_digits
        self.maximum_adjusted_numerator[index] = max(
            self.maximum_adjusted_numerator[index],
            numerator_digits - denominator_digits,
        )
        lifted_numerator = numerator_digits
        if self.slot_lcms is not None:
            slot_lcm = self.slot_lcms[index]
            if slot_lcm is None:
                return
            if slot_lcm != 1:
                if denominator is None:
                    raise ValueError(
                        "slot-LCM height sums require a source denominator"
                    )
                common = gcd(slot_lcm, denominator)
                lift = max(1, slot_lcm // common)
                lifted_numerator += _decimal_digits_from_bits(lift.bit_length())
        elif self.shared_lcm is not None:
            if denominator is None:
                raise ValueError("shared-LCM height sums require a source denominator")
            lift_digits = self._shared_lift_digits.get(denominator)
            if lift_digits is None:
                common = gcd(self.shared_lcm, denominator)
                lift = max(1, self.shared_lcm // common)
                lift_digits = _decimal_digits_from_bits(lift.bit_length())
                self._shared_lift_digits[denominator] = lift_digits
            lifted_numerator += lift_digits
        self.maximum_numerator[index] = max(
            self.maximum_numerator[index], lifted_numerator
        )
        self.term_counts[index] += 1

    def height(self, index: int) -> RationalHeight:
        count = self.term_counts[index]
        if count == 0:
            return RationalHeight(1, 1)
        if self.slot_lcms is not None:
            slot_lcm = self.slot_lcms[index]
            if slot_lcm is None:
                return RationalHeight(MAX_CANONICAL_RATIONAL_DIGITS + 1, 1)
            return RationalHeight(
                self.maximum_numerator[index] + len(str(count)),
                _decimal_digits_from_bits(slot_lcm.bit_length()),
            )
        if self.shared_lcm is not None:
            return RationalHeight(
                self.maximum_numerator[index] + len(str(count)),
                _decimal_digits_from_bits(self.shared_lcm.bit_length()),
            )
        denominator_digits = self.denominator_digits[index]
        return RationalHeight(
            self.maximum_adjusted_numerator[index]
            + denominator_digits
            + len(str(count)),
            denominator_digits,
        )

    def heights(self) -> tuple[RationalHeight, ...]:
        return tuple(self.height(index) for index in range(len(self.term_counts)))


def _heights(values: tuple[CanonicalRational, ...]) -> tuple[RationalHeight, ...]:
    return tuple(RationalHeight.from_canonical(value) for value in values)


def _require_result_height(height: RationalHeight, operation: str) -> None:
    if height.exceeds(MAX_CANONICAL_RATIONAL_DIGITS):
        raise OperationDomainValidationError(
            location=("values",),
            code="arithmetic_functions.result_height_exceeded",
            message=(
                f"{operation} rational height exceeds the "
                f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit result bound"
            ),
        )


def _require_result_envelope(
    heights: tuple[RationalHeight, ...], operation: str
) -> None:
    for height in heights:
        _require_result_height(height, operation)


def _require_length(
    values: tuple[CanonicalRational, ...], name: str, maximum: int
) -> None:
    if not 1 <= len(values) <= maximum:
        raise ValueError(f"{name} must have between 1 and {maximum} values")


def _decimal_digits_from_bits(bits: int) -> int:
    # 30103 / 100000 is a strict upper rational approximation to log10(2).
    return max(1, (bits * 30_103 + 99_999) // 100_000)


def _bounded_lcm(left: int, right: int) -> int | None:
    """Return ``lcm(left, right)`` when its digits stay in the result bound."""

    common = gcd(left, right)
    predicted_bits = left.bit_length() + right.bit_length() - common.bit_length() + 1
    if _decimal_digits_from_bits(predicted_bits) > MAX_CANONICAL_RATIONAL_DIGITS:
        return None
    return left // common * right


def _shared_denominator_lcm(
    values: tuple[CanonicalRational, ...],
) -> int | None:
    """Return ``lcm(dens)`` when its digits stay in the result bound."""

    if not values:
        return None
    running = parse_canonical_integer(values[0].den)
    seen = {values[0].den}
    for value in values[1:]:
        den = value.den
        if den in seen:
            continue
        seen.add(den)
        merged = _bounded_lcm(running, parse_canonical_integer(den))
        if merged is None:
            return None
        running = merged
    if _decimal_digits_from_bits(running.bit_length()) > MAX_CANONICAL_RATIONAL_DIGITS:
        return None
    return running


def _convolution_slot_lcms(
    length: int,
    incidences: tuple[tuple[int, int], ...],
    left_dens: tuple[int, ...],
    right_dens: tuple[int, ...],
) -> tuple[int | None, ...]:
    """Bound the denominator LCM independently for each output position."""

    cached_product = lru_cache(maxsize=4_096)(_bounded_product)
    cached_lcm = lru_cache(maxsize=4_096)(_bounded_lcm)
    lcms: list[int | None] = [1] * length
    interned_lcms: dict[int, int] = {1: 1}
    for divisor, multiple in incidences:
        index = multiple - 1
        current = lcms[index]
        if current is None:
            continue
        term_denominator = cached_product(
            left_dens[divisor - 1], right_dens[multiple // divisor - 1]
        )
        if term_denominator is None:
            lcms[index] = None
            continue
        merged = cached_lcm(current, term_denominator)
        if merged is not None:
            merged = interned_lcms.setdefault(merged, merged)
        lcms[index] = merged
    return tuple(lcms)


def _bounded_product(left: int, right: int) -> int | None:
    predicted_bits = left.bit_length() + right.bit_length()
    if _decimal_digits_from_bits(predicted_bits) > MAX_CANONICAL_RATIONAL_DIGITS:
        return None
    return left * right


def _admit_convolution(
    f: tuple[CanonicalRational, ...], g: tuple[CanonicalRational, ...]
) -> _ConvolutionPlan:
    _require_length(f, "f", _MAX_DIVISOR_PREFIX_LENGTH)
    if len(f) != len(g):
        raise ValueError("f and g must have the same length")
    incidences = _require_divisor_incidences(len(f))
    left = _heights(f)
    right = _heights(g)
    left_ratios = tuple(value.as_integer_ratio() for value in f)
    right_ratios = tuple(value.as_integer_ratio() for value in g)
    left_nums = tuple(numerator for numerator, _ in left_ratios)
    left_dens = tuple(denominator for _, denominator in left_ratios)
    right_nums = tuple(numerator for numerator, _ in right_ratios)
    right_dens = tuple(denominator for _, denominator in right_ratios)
    slot_lcms = _convolution_slot_lcms(len(left), incidences, left_dens, right_dens)
    sums = _HeightSums(len(left), slot_lcms=slot_lcms)
    for divisor, multiple in incidences:
        left_index = divisor - 1
        right_index = multiple // divisor - 1
        slot_lcm = slot_lcms[multiple - 1]
        sums.add_product(
            multiple - 1,
            left[left_index],
            right[right_index],
            denominator=(
                left_dens[left_index] * right_dens[right_index]
                if slot_lcm not in (None, 1)
                else None
            ),
        )
    _require_result_envelope(sums.heights(), "Dirichlet convolution")
    assert all(slot_lcm is not None for slot_lcm in slot_lcms)
    return _ConvolutionPlan(
        incidences=incidences,
        left_nums=left_nums,
        left_dens=left_dens,
        right_nums=right_nums,
        right_dens=right_dens,
        slot_lcms=tuple(slot_lcm for slot_lcm in slot_lcms if slot_lcm is not None),
    )


def _admit_mobius(
    values: tuple[CanonicalRational, ...],
) -> _MobiusPlan:
    _require_length(values, "values", _MAX_DIVISOR_PREFIX_LENGTH)
    incidences = _require_divisor_incidences(len(values))
    heights = _heights(values)
    ratios = tuple(
        (parse_canonical_integer(value.num), parse_canonical_integer(value.den))
        for value in values
    )
    dens = tuple(denominator for _, denominator in ratios)
    shared_lcm = _shared_denominator_lcm(values)
    sums = _HeightSums(
        len(heights),
        shared_lcm=shared_lcm,
    )
    for divisor, multiple in incidences:
        source = multiple // divisor - 1
        sums.add(multiple - 1, heights[source], denominator=dens[source])
    _require_result_envelope(sums.heights(), "Möbius transform")
    return _MobiusPlan(
        incidences=incidences,
        numerators=tuple(numerator for numerator, _ in ratios),
        denominators=dens,
        shared_lcm=shared_lcm,
    )


def _admit_inverse(values: tuple[CanonicalRational, ...]) -> None:
    _require_length(values, "values", _MAX_DIVISOR_PREFIX_LENGTH)
    if values[0].as_fraction() == 0:
        raise ValueError("f(1) must be nonzero")
    _require_divisor_incidences(len(values))
    source = _heights(values)
    if (
        all(
            height.numerator_digits <= 1 and height.denominator_digits <= 1
            for height in source
        )
        and source[0].numerator_digits == 1
    ):
        # Unit-height prefixes invert to Möbius-scale values (±1, 0, 1).
        _require_result_envelope(
            tuple(RationalHeight(1, 1) for _ in source), "Dirichlet inverse"
        )
        return
    inverse = [RationalHeight(1, 1)] * len(source)
    inverse[0] = RationalHeight(1, 1).quotient(source[0])
    sums = _HeightSums(len(source))
    for quotient in range(1, len(source) + 1):
        if quotient > 1:
            height = sums.height(quotient - 1).quotient(source[0])
            inverse[quotient - 1] = height
        else:
            height = inverse[0]
        _require_result_height(height, "Dirichlet inverse")
        for divisor in range(2, len(source) // quotient + 1):
            sums.add(
                quotient * divisor - 1,
                source[divisor - 1].product(height),
            )
    _require_result_envelope(tuple(inverse), "Dirichlet inverse")


def _mobius_sieve(n: int) -> list[int]:
    """Return ``mu`` for 1..n where ``mu`` is the Möbius function.

    ``mu(1) = 1``, and for ``k > 1`` ``mu(k) = 0`` if ``k`` has a squared
    prime factor, otherwise ``(-1)^(number of distinct prime factors)``.
    """
    mobius = [0] * (n + 1)
    if n >= 1:
        mobius[1] = 1
    is_prime = [True] * (n + 1)
    smallest_prime_factor = [0] * (n + 1)
    for i in range(2, n + 1):
        if is_prime[i]:
            smallest_prime_factor[i] = i
            for j in range(i * i, n + 1, i):
                if is_prime[j]:
                    is_prime[j] = False
                    smallest_prime_factor[j] = i
        p = smallest_prime_factor[i]
        if p == i:
            # i is prime
            mobius[i] = -1
        else:
            # i = i // p; check if p divides (i // p)
            reduced = i // p
            if reduced % p == 0:
                mobius[i] = 0
            else:
                mobius[i] = -mobius[reduced]
    return mobius


def dirichlet_convolution(
    f: tuple[CanonicalRational, ...], g: tuple[CanonicalRational, ...]
) -> tuple[CanonicalRational, ...]:
    """Compute ``h = f * g`` where ``h(K) = sum_{d|K} f(d) * g(K/d)``."""
    plan = _admit_convolution(f, g)
    n = len(f)
    result_numerators = [0] * n
    for divisor, multiple in plan.incidences:
        left_index = divisor - 1
        right_index = multiple // divisor - 1
        denominator = plan.left_dens[left_index] * plan.right_dens[right_index]
        result_numerators[multiple - 1] += (
            plan.left_nums[left_index]
            * plan.right_nums[right_index]
            * (plan.slot_lcms[multiple - 1] // denominator)
        )
    return tuple(
        CanonicalRational.from_integer_ratio(numerator, denominator)
        for numerator, denominator in zip(
            result_numerators, plan.slot_lcms, strict=True
        )
    )


def mobius_transform(
    values: tuple[CanonicalRational, ...], inverse: bool = False
) -> tuple[CanonicalRational, ...]:
    """Compute the Möbius (inverse) transform.

    Forward:  ``f(K) = sum_{d|K} mu(d) * F(K/d)`` (input is F, output is f).
    Inverse: ``F(K) = sum_{d|K} f(K/d)``           (input is f, output is F).

    The forward transform is Dirichlet convolution with the Möbius function
    ``mu`` (``f = mu * F``); the inverse is Dirichlet convolution with the
    constant-one function ``1`` (``F = 1 * f``), since ``mu * 1 = epsilon``.
    The two operations are mutually inverse: forward then inverse (or vice
    versa) recovers the original function.
    """
    plan = _admit_mobius(values)
    n = len(values)
    if plan.shared_lcm is not None:
        result_numerators = [0] * n
        mobius = None if inverse else _mobius_sieve(n)
        for divisor, multiple in plan.incidences:
            coefficient = 1 if mobius is None else mobius[divisor]
            if coefficient:
                source = multiple // divisor - 1
                result_numerators[multiple - 1] += (
                    coefficient
                    * plan.numerators[source]
                    * (plan.shared_lcm // plan.denominators[source])
                )
        return tuple(
            CanonicalRational.from_integer_ratio(numerator, plan.shared_lcm)
            for numerator in result_numerators
        )

    fraction_values = [v.as_fraction() for v in values]
    result_values: list[Fraction] = [Fraction(0)] * n
    mobius = None if inverse else _mobius_sieve(n)
    for divisor, multiple in plan.incidences:
        coefficient = 1 if mobius is None else mobius[divisor]
        if coefficient:
            result_values[multiple - 1] += (
                coefficient * fraction_values[multiple // divisor - 1]
            )
    return tuple(_rational(value) for value in result_values)


def summatory_function(
    values: tuple[CanonicalRational, ...],
) -> tuple[CanonicalRational, ...]:
    """Compute ``S(K) = sum_{i=1}^{K} f(i)`` for K = 1..n."""
    _require_length(values, "values", _MAX_SUMMATORY_LENGTH)
    _require_result_height(sum_heights(_heights(values)), "summatory function")
    n = len(values)
    fraction_values = [v.as_fraction() for v in values]
    result_values: list[Fraction] = [Fraction(0)] * n
    running = Fraction(0)
    for i in range(n):
        running += fraction_values[i]
        result_values[i] = running
    return tuple(_rational(v) for v in result_values)


def dirichlet_inverse(
    values: tuple[CanonicalRational, ...],
) -> tuple[CanonicalRational, ...]:
    """Compute the Dirichlet inverse ``g`` of ``f`` such that ``f * g = epsilon``.

    The Dirichlet inverse is defined recursively:
    ``g(1) = 1 / f(1)`` and for ``K > 1``:
    ``g(K) = -(1 / f(1)) * sum_{d | K, d > 1} f(d) * g(K / d)``.

    The first element of ``f`` (i.e. ``f(1)``) must be non-zero.
    """
    _admit_inverse(values)
    f = [v.as_fraction() for v in values]
    n = len(values)
    g: list[Fraction] = [Fraction(0)] * n
    g[0] = Fraction(1) / f[0]
    partials = [Fraction(0)] * n
    for quotient in range(1, n + 1):
        if quotient > 1:
            g[quotient - 1] = -(partials[quotient - 1] / f[0])
        for divisor in range(2, n // quotient + 1):
            partials[quotient * divisor - 1] += f[divisor - 1] * g[quotient - 1]
    return tuple(_rational(v) for v in g)


__all__ = [
    "dirichlet_convolution",
    "dirichlet_inverse",
    "mobius_transform",
    "summatory_function",
]
