"""Supported native APIs for exact classical combinatorial numbers."""

from __future__ import annotations

from fractions import Fraction


def _nonnegative(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _pair(n: int, k: int) -> tuple[int, int]:
    return _nonnegative(n, name="n"), _nonnegative(k, name="k")


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
    "catalan_number",
    "derangement_number",
    "double_factorial",
    "fibonacci_number",
    "integer_partitions",
    "lucas_number",
    "motzkin_number",
    "partition_number",
    "stirling_first",
    "stirling_second",
]
