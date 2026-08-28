"""Native arithmetic-function operations over canonical rational values."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math._rational_height import RationalHeight, sum_heights
from jacobian.math.number_theory.arithmetic_functions._models import (
    _MAX_LENGTH,
)


def _rational(value: Fraction | int) -> CanonicalRational:
    """Convert an exact Fraction or int to a CanonicalRational."""
    return CanonicalRational.from_fraction(Fraction(value))


def _divisors(n: int) -> list[int]:
    """Return the divisors of ``n`` in increasing order."""
    if n <= 0:
        return []
    small: list[int] = []
    large: list[int] = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    return small + large[::-1]


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


def _require_length(values: tuple[CanonicalRational, ...], name: str) -> None:
    if not 1 <= len(values) <= _MAX_LENGTH:
        raise ValueError(f"{name} must have between 1 and {_MAX_LENGTH} values")


def _admit_convolution(
    f: tuple[CanonicalRational, ...], g: tuple[CanonicalRational, ...]
) -> None:
    _require_length(f, "f")
    if len(f) != len(g):
        raise ValueError("f and g must have the same length")
    left = _heights(f)
    right = _heights(g)
    for index in range(1, len(left) + 1):
        terms = (
            left[divisor - 1].product(right[index // divisor - 1])
            for divisor in _divisors(index)
        )
        _require_result_height(sum_heights(terms), "Dirichlet convolution")


def _admit_mobius(values: tuple[CanonicalRational, ...]) -> None:
    _require_length(values, "values")
    heights = _heights(values)
    for index in range(1, len(heights) + 1):
        terms = (heights[index // divisor - 1] for divisor in _divisors(index))
        _require_result_height(sum_heights(terms), "Möbius transform")


def _admit_inverse(values: tuple[CanonicalRational, ...]) -> None:
    _require_length(values, "values")
    if values[0].as_fraction() == 0:
        raise ValueError("f(1) must be nonzero")
    source = _heights(values)
    inverse = [RationalHeight(1, 1).quotient(source[0])]
    _require_result_height(inverse[0], "Dirichlet inverse")
    for index in range(2, len(source) + 1):
        terms = (
            source[divisor - 1].product(inverse[index // divisor - 1])
            for divisor in _divisors(index)
            if divisor > 1
        )
        height = sum_heights(terms).quotient(source[0])
        _require_result_height(height, "Dirichlet inverse")
        inverse.append(height)


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
    # Precompute divisors for each k = 1..n.
    result_values: list[Fraction] = [Fraction(0)] * n
    for k in range(1, n + 1):
        acc = Fraction(0)
        for d in _divisors(k):
            acc += f_values[d - 1] * g_values[k // d - 1]
        result_values[k - 1] = acc
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
        for k in range(1, n + 1):
            acc = Fraction(0)
            for d in _divisors(k):
                acc += fraction_values[k // d - 1]
            result_values[k - 1] = acc
    else:
        # f(K) = sum_{d|K} mu(d) * F(K/d)  (Dirichlet convolution with mu)
        mobius = _mobius_sieve(n)
        for k in range(1, n + 1):
            acc = Fraction(0)
            for d in _divisors(k):
                if mobius[d] == 0:
                    continue
                acc += mobius[d] * fraction_values[k // d - 1]
            result_values[k - 1] = acc
    return tuple(_rational(v) for v in result_values)


def summatory_function(
    values: tuple[CanonicalRational, ...],
) -> tuple[CanonicalRational, ...]:
    """Compute ``S(K) = sum_{i=1}^{K} f(i)`` for K = 1..n."""
    _require_length(values, "values")
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
    for k in range(2, n + 1):
        divs = _divisors(k)
        partial = Fraction(0)
        for d in divs:
            if d == 1:
                continue
            partial += f[d - 1] * g[k // d - 1]
        g[k - 1] = -(partial / f[0])
    return tuple(_rational(v) for v in g)


__all__ = [
    "dirichlet_convolution",
    "dirichlet_inverse",
    "mobius_transform",
    "summatory_function",
]
