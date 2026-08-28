"""Supported native APIs for exact classical combinatorial numbers."""

from __future__ import annotations

from fractions import Fraction


def _nonnegative(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _pair(n: int, k: int) -> tuple[int, int]:
    return _nonnegative(n, name="n"), _nonnegative(k, name="k")


MAX_COUNTING_INDEX = 10_000
MAX_MULTINOMIAL_PARTS = 256
MAX_MULTINOMIAL_TOTAL = MAX_COUNTING_INDEX


def _bounded_counting_index(value: int, *, name: str) -> int:
    value = _nonnegative(value, name=name)
    if value > MAX_COUNTING_INDEX:
        raise ValueError(
            f"{name} exceeds the {MAX_COUNTING_INDEX}-element counting bound"
        )
    return value


def factorial(n: int) -> int:
    """Return the factorial of a bounded nonnegative integer."""
    import math

    return math.factorial(_bounded_counting_index(n, name="n"))


def binomial(n: int, k: int) -> int:
    """Return the exact binomial coefficient, with zero for ``k > n``."""
    import math

    first = _bounded_counting_index(n, name="n")
    second = _bounded_counting_index(k, name="k")
    return 0 if second > first else math.comb(first, second)


def multinomial(values: tuple[int, ...]) -> int:
    """Return the exact multinomial coefficient for nonnegative part sizes."""
    import math

    if not isinstance(values, tuple) or not values:
        raise ValueError("values must be a nonempty tuple of nonnegative integers")
    parts = tuple(_nonnegative(value, name="values") for value in values)
    if len(parts) > MAX_MULTINOMIAL_PARTS:
        raise ValueError(
            f"values exceeds the {MAX_MULTINOMIAL_PARTS}-part counting bound"
        )
    if len(parts) == 1:
        return 1
    total = sum(parts)
    if total > MAX_MULTINOMIAL_TOTAL:
        raise ValueError(
            "the sum of values exceeds the "
            f"{MAX_MULTINOMIAL_TOTAL}-element counting bound"
        )
    return math.factorial(total) // math.prod(math.factorial(part) for part in parts)


def permutations(n: int, k: int) -> int:
    """Return the exact number of ordered ``k``-selections from ``n``."""
    import math

    first = _bounded_counting_index(n, name="n")
    second = _bounded_counting_index(k, name="k")
    return 0 if second > first else math.perm(first, second)


def central_binomial(n: int) -> int:
    """Return the exact central binomial coefficient ``binomial(2n, n)``."""
    import math

    value = _bounded_counting_index(n, name="n")
    return math.comb(2 * value, value)


def compositions(n: int, k: int) -> int:
    """Count ordered compositions of ``n`` into ``k`` positive parts."""
    import math

    total = _bounded_counting_index(n, name="n")
    parts = _bounded_counting_index(k, name="k")
    if total == parts == 0:
        return 1
    return math.comb(total - 1, parts - 1) if 0 < parts <= total else 0


def bell_number(n: int) -> int:
    """Return the nth Bell number."""

    import sympy

    return int(sympy.bell(_nonnegative(n, name="n")))


def bernoulli_number(n: int) -> Fraction:
    """Return the nth Bernoulli number exactly."""

    import sympy

    value = sympy.bernoulli(_nonnegative(n, name="n"))
    return Fraction(int(value.p), int(value.q))


def catalan_number(n: int) -> int:
    """Return the nth Catalan number."""

    import sympy

    return int(sympy.catalan(_nonnegative(n, name="n")))


def derangement_number(n: int) -> int:
    """Return the number of derangements of n objects."""

    import sympy

    return int(sympy.subfactorial(_nonnegative(n, name="n")))


def double_factorial(n: int) -> int:
    """Return the nonnegative integer double factorial."""

    import sympy

    return int(sympy.factorial2(_nonnegative(n, name="n")))


def fibonacci_number(n: int) -> int:
    """Return the nth Fibonacci number."""

    import sympy

    return int(sympy.fibonacci(_nonnegative(n, name="n")))


def integer_partitions(
    n: int,
    *,
    max_parts: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate integer partitions in deterministic reverse-part order."""

    from sympy.utilities.iterables import partitions

    value = _nonnegative(n, name="n")
    if max_parts is not None:
        max_parts = _nonnegative(max_parts, name="max_parts")
    return tuple(
        tuple(
            part
            for part in sorted(multiplicities, reverse=True)
            for _ in range(int(multiplicities[part]))
        )
        for multiplicities in partitions(value, m=max_parts)
    )


def lucas_number(n: int) -> int:
    """Return the nth Lucas number."""

    import sympy

    return int(sympy.lucas(_nonnegative(n, name="n")))


def motzkin_number(n: int) -> int:
    """Return the nth Motzkin number."""

    import sympy

    return int(sympy.motzkin(_nonnegative(n, name="n")))


def partition_number(n: int) -> int:
    """Return the number of integer partitions of n."""

    import sympy

    return int(sympy.partition(_nonnegative(n, name="n")))


def stirling_first(n: int, k: int) -> int:
    """Return the unsigned Stirling number of the first kind."""

    from sympy.functions.combinatorial.numbers import stirling

    first, second = _pair(n, k)
    return int(stirling(first, second, kind=1))


def stirling_second(n: int, k: int) -> int:
    """Return the Stirling number of the second kind."""

    from sympy.functions.combinatorial.numbers import stirling

    first, second = _pair(n, k)
    return int(stirling(first, second, kind=2))


__all__ = [
    "bell_number",
    "bernoulli_number",
    "binomial",
    "catalan_number",
    "central_binomial",
    "compositions",
    "derangement_number",
    "double_factorial",
    "factorial",
    "fibonacci_number",
    "integer_partitions",
    "lucas_number",
    "motzkin_number",
    "multinomial",
    "partition_number",
    "permutations",
    "stirling_first",
    "stirling_second",
]
