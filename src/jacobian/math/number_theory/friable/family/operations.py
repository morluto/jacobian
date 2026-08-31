"""Exact bounded friable-family enumeration kernel."""

from __future__ import annotations

from math import isqrt

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.arithmetic.values import IntegerValue
from jacobian.math.number_theory.friable.family._models import (
    plan_friable_family,
)


def _family_materialized(x: int, y: int) -> list[int]:
    """Collect every y-friable integer in [1, x] using a local sieve."""

    is_prime = bytearray(b"\x01") * (x + 1)
    is_prime[0:2] = b"\x00\x00"
    is_friable = bytearray(b"\x01") * (x + 1)
    is_friable[0] = 0
    square_root = isqrt(x)

    for prime in range(2, x + 1):
        if not is_prime[prime]:
            continue
        if prime <= square_root:
            first_composite = prime * prime
            composite_count = (x - first_composite) // prime + 1
            is_prime[first_composite : x + 1 : prime] = b"\x00" * composite_count
        if prime > y:
            is_friable[prime : x + 1 : prime] = b"\x00" * (x // prime)

    return [n for n in range(1, x + 1) if is_friable[n]]


def _family_generated(x: int, primes: tuple[int, ...]) -> list[int]:
    """Enumerate prime-exponent vectors and collect their products.

    Each prime-exponent vector whose product is at most x corresponds to a
    unique y-friable integer.  We recursively generate products, then sort and
    deduplicate to produce the canonical increasing family.
    """

    products: list[int] = []

    def visit(prime_index: int, current: int) -> None:
        if prime_index == len(primes):
            products.append(current)
            return
        prime = primes[prime_index]
        value = current
        while True:
            visit(prime_index + 1, value)
            if value > x // prime:
                return
            value *= prime

    visit(0, 1)
    products.sort()
    return products


def enumerate_friable_family_kernel(x: int, y: int) -> list[int]:
    """Return the increasing list of positive y-friable integers at most x.

    The cutoff is inclusive. By convention ``Psi-family(0, y) = []`` and, for
    positive ``x``, ``Psi-family(x, 0) = Psi-family(x, 1) = [1]``.
    """
    regime, primes = plan_friable_family(x, y)
    if regime == "DIRECT":
        if x == 0:
            return []
        if y <= 1:
            return [1]
        return list(range(1, x + 1))
    if regime == "MATERIALIZED":
        return _family_materialized(x, y)
    return _family_generated(x, primes)


def enumerate_friable_family(x: int | IntegerValue, y: int | IntegerValue) -> list[int]:
    """Return the complete ordered tuple of y-friable integers at most x.

    This is the family analogue of :func:`count_friable`.  The result is one
    source-bound finite family; callers may form sumsets, gaps, or outer
    searches from it.
    """
    if isinstance(x, IntegerValue) and isinstance(y, IntegerValue):
        x_text = x.value
        y_text = y.value
        if x_text == "0" or y_text in {"0", "1"}:
            return [] if x_text == "0" else [1]
        for text in (x_text, y_text):
            if len(text.lstrip("-")) > 256:
                raise OperationDomainValidationError(
                    location=("x", "y"),
                    code="number_theory.friable_family.source_digit_bound",
                    message="friable-family sources must have at most 256 decimal digits",
                )
    x_val = _as_python_integer(x)
    y_val = _as_python_integer(y)
    if type(x_val) is not int or type(y_val) is not int:
        raise TypeError("friable-family inputs must be integers")
    return enumerate_friable_family_kernel(x_val, y_val)


def _as_python_integer(value: int | IntegerValue) -> int:
    """Return one admitted integer input as its Python integer value."""

    if isinstance(value, IntegerValue):
        return parse_canonical_integer(value.value)
    return value


__all__ = ["enumerate_friable_family"]
