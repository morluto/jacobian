"""Exact arithmetic counting operations."""

from math import gcd

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.counting._models import (
    _MAX_BOX_COORD,
    _MAX_BOX_LINEAR_COEFFICIENT,
    _MAX_BOX_MODULUS,
    _MAX_FLOOR_SUM_N,
    _MAX_FLOOR_SUM_PARAM,
)


def _reject(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=f"arithmetic_counting.{code}",
        message=message,
    )


def floor_sum(n: int, m: int, a: int, b: int) -> int:
    """Return ``sum(floor((a*i+b)/m) for i in range(n))`` exactly."""

    if not 0 <= n <= _MAX_FLOOR_SUM_N:
        _reject(("n",), "n_out_of_range", "n is outside the admitted range")
    for name, value in (("a", a), ("b", b)):
        if not 0 <= value <= _MAX_FLOOR_SUM_PARAM:
            _reject(
                (name,),
                f"{name}_out_of_range",
                f"{name} is outside the admitted range",
            )
    if not 1 <= m <= _MAX_FLOOR_SUM_PARAM:
        _reject(("m",), "m_out_of_range", "m is outside the admitted range")

    answer = 0
    while True:
        if a >= m:
            answer += (n - 1) * n // 2 * (a // m)
            a %= m
        if b >= m:
            answer += n * (b // m)
            b %= m
        maximum = a * n + b
        if maximum < m:
            return answer
        n = maximum // m
        b = maximum % m
        m, a = a, m


def congruence_box_count(
    *,
    x_lo: int,
    x_hi: int,
    y_lo: int,
    y_hi: int,
    u: int,
    v: int,
    c: int,
    modulus: int,
) -> int:
    """Count points in a bounded box satisfying one linear congruence."""

    for name, value in (
        ("x_lo", x_lo),
        ("x_hi", x_hi),
        ("y_lo", y_lo),
        ("y_hi", y_hi),
    ):
        if not -_MAX_BOX_COORD <= value <= _MAX_BOX_COORD:
            _reject(
                (name,),
                f"{name}_out_of_range",
                f"{name} is outside the admitted range",
            )
    if x_lo > x_hi:
        _reject(("x_lo", "x_hi"), "x_interval_invalid", "x_lo must be <= x_hi")
    if y_lo > y_hi:
        _reject(("y_lo", "y_hi"), "y_interval_invalid", "y_lo must be <= y_hi")
    if not 1 <= modulus <= _MAX_BOX_MODULUS:
        _reject(
            ("modulus",),
            "modulus_out_of_range",
            "modulus is outside the admitted range",
        )
    for name, value in (("u", u), ("v", v), ("c", c)):
        if not -_MAX_BOX_LINEAR_COEFFICIENT <= value <= _MAX_BOX_LINEAR_COEFFICIENT:
            _reject(
                (name,),
                f"{name}_out_of_range",
                f"{name} is outside the admitted range",
            )

    x_length = x_hi - x_lo + 1
    y_length = y_hi - y_lo + 1
    if y_length < x_length:
        x_lo, x_hi, y_lo, y_hi = y_lo, y_hi, x_lo, x_hi
        u, v = v, u

    divisor = gcd(v, modulus)
    reduced_modulus = modulus // divisor
    inverse = 0 if reduced_modulus == 1 else pow(v // divisor, -1, reduced_modulus)

    count = 0
    for x in range(x_lo, x_hi + 1):
        right_hand_side = c - u * x
        if right_hand_side % divisor:
            continue
        residue = (right_hand_side // divisor * inverse) % reduced_modulus
        count += (y_hi - residue) // reduced_modulus
        count -= (y_lo - 1 - residue) // reduced_modulus
    return count


__all__ = ["congruence_box_count", "floor_sum"]
