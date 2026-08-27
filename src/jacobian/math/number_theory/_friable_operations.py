"""Exact bounded friable-count kernel and typed operation adapter."""

from __future__ import annotations

from math import isqrt

from pydantic import ValidationError

from jacobian.canonical import parse_canonical_integer
from jacobian.math.arithmetic.values import IntegerValue
from jacobian.math.number_theory._friable_models import (
    FriableCountRequest,
    FriableCountResult,
    _plan_friable_count,
)


def _count_materialized(x: int, y: int) -> int:
    """Count integers with no prime factor above ``y`` using local sieves."""

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

    return sum(is_friable)


def _count_generated(x: int, primes: tuple[int, ...]) -> int:
    """Count bounded prime-exponent tuples without materializing ``1..x``."""

    def visit(prime_index: int, remaining: int) -> int:
        if prime_index == len(primes):
            return 1
        prime = primes[prime_index]
        total = 0
        while True:
            total += visit(prime_index + 1, remaining)
            if remaining < prime:
                return total
            remaining //= prime

    return visit(0, x)


def _as_python_integer(value: int | IntegerValue) -> int:
    """Return one admitted integer input as its Python integer value."""

    if isinstance(value, IntegerValue):
        return parse_canonical_integer(value.value)
    return value


def count_friable(x: int | IntegerValue, y: int | IntegerValue) -> int:
    """Return ``Psi(x, y)``, the number of positive y-friable integers <= x.

    The cutoff is inclusive. By convention ``Psi(0, y) = 0`` and, for
    positive ``x``, ``Psi(x, 0) = Psi(x, 1) = 1``. The function accepts the
    same result-sensitive execution envelope as the public operation.
    """

    x = _as_python_integer(x)
    y = _as_python_integer(y)
    if type(x) is not int or type(y) is not int:
        raise TypeError("friable-count inputs must be integers")
    regime, primes = _plan_friable_count(x, y)
    if regime == "DIRECT":
        if x == 0:
            return 0
        if y <= 1:
            return 1
        return x
    if regime == "MATERIALIZED":
        return _count_materialized(x, y)
    return _count_generated(x, primes)


def compute_friable_count(request: FriableCountRequest) -> FriableCountResult:
    """Compute one exact source-bound friable count."""

    x = parse_canonical_integer(request.x)
    y = parse_canonical_integer(request.y)
    return FriableCountResult._from_kernel(
        request,
        count=count_friable(x, y),
    )


def verify_friable_count_result(result: FriableCountResult) -> bool:
    """Check an independently supplied exact count inside the owner envelope."""

    try:
        request = FriableCountRequest(x=result.x, y=result.y)
    except ValidationError:
        return False
    return parse_canonical_integer(result.count) == count_friable(
        parse_canonical_integer(request.x), parse_canonical_integer(request.y)
    )


__all__ = ["compute_friable_count", "count_friable"]
