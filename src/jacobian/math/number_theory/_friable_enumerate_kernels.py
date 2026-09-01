"""Exact bounded kernel for friable-family enumeration."""

from __future__ import annotations

from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory._friable_enumerate_models import (
    plan_friable_enumerate,
)
from jacobian.math.number_theory.arithmetic.values import IntegerValue


def _enumerate_materialized(x: int, y: int) -> tuple[int, ...]:
    """Return the increasing tuple of y-friable integers at most x.

    A direct sieve marks integers in 1..x whose greatest prime factor exceeds y.
    """

    is_friable = bytearray(b"\x01") * (x + 1)
    is_friable[0] = 0

    # Mark non-friable integers: for each prime > y, mark its multiples.
    # We use a sieve-of-Eratosthenes-like pass to find primes and mark composites.
    is_prime = bytearray(b"\x01") * (x + 1)
    is_prime[0:2] = b"\x00\x00"
    for candidate in range(2, x + 1):
        if not is_prime[candidate]:
            continue
        # Sieve for primes
        if candidate * candidate <= x:
            first = candidate * candidate
            is_prime[first : x + 1 : candidate] = b"\x00" * (
                (x - first) // candidate + 1
            )
        # If this prime exceeds y, all its multiples are non-friable
        if candidate > y:
            is_friable[candidate : x + 1 : candidate] = b"\x00" * (x // candidate)

    return tuple(i for i in range(1, x + 1) if is_friable[i])


def _enumerate_generated(x: int, primes: tuple[int, ...]) -> tuple[int, ...]:
    """Generate y-friable integers via exponent-vector recursion, then sort.

    The recursion builds products: for each prime, it tries exponent 0, 1, 2,
    ... as long as the running product stays at most x.  At the leaf (all
    primes exhausted), the accumulated product is one y-friable integer.
    """

    if not primes:
        return ()

    values: list[int] = []

    def visit(prime_index: int, product: int) -> None:
        if prime_index == len(primes):
            values.append(product)
            return
        prime = primes[prime_index]
        while product <= x:
            visit(prime_index + 1, product)
            product *= prime

    visit(0, 1)
    values.sort()
    return tuple(values)


def enumerate_friable(
    x: int | IntegerValue,
    y: int | IntegerValue,
) -> tuple[int, ...]:
    """Return the increasing tuple of positive y-friable integers at most x.

    The cutoff is inclusive. By convention, for positive x and y <= 1, only 1
    is friable. For x = 0, the family is empty.
    """

    x = _as_python_integer(x)
    y = _as_python_integer(y)
    if type(x) is not int or type(y) is not int:
        raise TypeError("friable-enumerate inputs must be integers")

    regime, _primes, admitted_family = plan_friable_enumerate(x, y)

    if regime == "DIRECT":
        return admitted_family

    if regime == "MATERIALIZED":
        return admitted_family
    return admitted_family


def _as_python_integer(value: int | IntegerValue) -> int:
    """Return one admitted integer input as its Python integer value."""

    if isinstance(value, IntegerValue):
        return parse_canonical_integer(value.value)
    return value


__all__ = ["enumerate_friable"]
