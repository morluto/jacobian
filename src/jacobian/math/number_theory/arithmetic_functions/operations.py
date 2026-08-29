"""Native arithmetic-function operations over canonical rational values."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math._rational_height import RationalHeight, sum_heights
from jacobian.math.number_theory.arithmetic_functions._models import (
    _MAX_DIVISOR_PREFIX_LENGTH,
    _MAX_SUMMATORY_LENGTH,
)

MAX_DIVISOR_INCIDENCES = 600_000
_RESULT_ENTRY_RESERVE_BYTES = 32
_RESULT_RESERVE_BYTES = 256


def _rational(value: Fraction | int) -> CanonicalRational:
    """Convert an exact Fraction or int to a CanonicalRational."""
    return CanonicalRational.from_fraction(Fraction(value))


def _divisor_incidences(
    length: int, *, minimum_divisor: int = 1
) -> Iterator[tuple[int, int]]:
    """Yield ``(divisor, multiple)`` for the bounded dense prefix."""
    for divisor in range(minimum_divisor, length + 1):
        for multiple in range(divisor, length + 1, divisor):
            yield divisor, multiple


def _divisor_incidence_count(length: int, *, minimum_divisor: int = 1) -> int:
    return sum(length // divisor for divisor in range(minimum_divisor, length + 1))


def _require_divisor_work(length: int) -> None:
    incidences = _divisor_incidence_count(length)
    if incidences > MAX_DIVISOR_INCIDENCES:
        raise OperationDomainValidationError(
            location=("values",),
            code="arithmetic_functions.divisor_incidence_work_exceeded",
            message=(
                "arithmetic-function prefix exceeds the "
                f"{MAX_DIVISOR_INCIDENCES}-incidence divisor-sieve budget"
            ),
        )


class _HeightSums:
    """Incremental form of ``sum_heights`` for one dense result vector."""

    def __init__(
        self, length: int, *, shared_denominator_digits: int | None = None
    ) -> None:
        self.denominator_digits = [0] * length
        self.maximum_adjusted_numerator = [0] * length
        self.maximum_numerator = [0] * length
        self.term_counts = [0] * length
        self.shared_denominator_digits = shared_denominator_digits

    def add(self, index: int, height: RationalHeight) -> None:
        self.denominator_digits[index] += height.denominator_digits
        self.maximum_adjusted_numerator[index] = max(
            self.maximum_adjusted_numerator[index],
            height.numerator_digits - height.denominator_digits,
        )
        self.maximum_numerator[index] = max(
            self.maximum_numerator[index], height.numerator_digits
        )
        self.term_counts[index] += 1

    def height(self, index: int) -> RationalHeight:
        count = self.term_counts[index]
        if count == 0:
            return RationalHeight(1, 1)
        if self.shared_denominator_digits is not None:
            return RationalHeight(
                self.maximum_numerator[index] + len(str(count)),
                self.shared_denominator_digits,
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
    predicted_bytes = _RESULT_RESERVE_BYTES + sum(
        height.numerator_digits
        + height.denominator_digits
        + _RESULT_ENTRY_RESERVE_BYTES
        for height in heights
    )
    if predicted_bytes > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("values",),
            code="arithmetic_functions.result_bytes_exceeded",
            message=(f"{operation} result exceeds the canonical output-byte limit"),
        )


def _require_length(
    values: tuple[CanonicalRational, ...], name: str, maximum: int
) -> None:
    if not 1 <= len(values) <= maximum:
        raise ValueError(f"{name} must have between 1 and {maximum} values")


def _admit_convolution(
    f: tuple[CanonicalRational, ...], g: tuple[CanonicalRational, ...]
) -> None:
    _require_length(f, "f", _MAX_DIVISOR_PREFIX_LENGTH)
    if len(f) != len(g):
        raise ValueError("f and g must have the same length")
    _require_divisor_work(len(f))
    left = _heights(f)
    right = _heights(g)
    sums = _HeightSums(len(left))
    for divisor, multiple in _divisor_incidences(len(left)):
        sums.add(
            multiple - 1,
            left[divisor - 1].product(right[multiple // divisor - 1]),
        )
    _require_result_envelope(sums.heights(), "Dirichlet convolution")


def _admit_mobius(values: tuple[CanonicalRational, ...]) -> None:
    _require_length(values, "values", _MAX_DIVISOR_PREFIX_LENGTH)
    _require_divisor_work(len(values))
    heights = _heights(values)
    shared_denominator_digits = (
        len(values[0].den)
        if all(value.den == values[0].den for value in values)
        else None
    )
    sums = _HeightSums(
        len(heights), shared_denominator_digits=shared_denominator_digits
    )
    for divisor, multiple in _divisor_incidences(len(heights)):
        sums.add(multiple - 1, heights[multiple // divisor - 1])
    _require_result_envelope(sums.heights(), "Möbius transform")


def _admit_inverse(values: tuple[CanonicalRational, ...]) -> None:
    _require_length(values, "values", _MAX_DIVISOR_PREFIX_LENGTH)
    if values[0].as_fraction() == 0:
        raise ValueError("f(1) must be nonzero")
    _require_divisor_work(len(values))
    source = _heights(values)
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
    _admit_convolution(f, g)
    n = len(f)
    f_values = [v.as_fraction() for v in f]
    g_values = [v.as_fraction() for v in g]
    result_values: list[Fraction] = [Fraction(0)] * n
    for divisor, multiple in _divisor_incidences(n):
        result_values[multiple - 1] += (
            f_values[divisor - 1] * g_values[multiple // divisor - 1]
        )
    return tuple(_rational(v) for v in result_values)


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
    _admit_mobius(values)
    n = len(values)
    fraction_values = [v.as_fraction() for v in values]
    result_values: list[Fraction] = [Fraction(0)] * n
    if inverse:
        # F(K) = sum_{d|K} f(K/d)  (Dirichlet convolution with 1)
        for divisor, multiple in _divisor_incidences(n):
            result_values[multiple - 1] += fraction_values[multiple // divisor - 1]
    else:
        # f(K) = sum_{d|K} mu(d) * F(K/d)  (Dirichlet convolution with mu)
        mobius = _mobius_sieve(n)
        for divisor, multiple in _divisor_incidences(n):
            if mobius[divisor]:
                result_values[multiple - 1] += (
                    mobius[divisor] * fraction_values[multiple // divisor - 1]
                )
    return tuple(_rational(v) for v in result_values)


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
