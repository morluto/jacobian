"""Exact arithmetic counting operations."""

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.counting._models import (
    _MAX_BOX_AREA,
    _MAX_BOX_COORD,
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
    area = (x_hi - x_lo + 1) * (y_hi - y_lo + 1)
    if area > _MAX_BOX_AREA:
        _reject(
            ("x_lo", "x_hi", "y_lo", "y_hi"),
            "box_area_exceeds_budget",
            "box area exceeds the computational budget",
        )
    return sum(
        (u * x + v * y - c) % modulus == 0
        for x in range(x_lo, x_hi + 1)
        for y in range(y_lo, y_hi + 1)
    )


__all__ = ["congruence_box_count", "floor_sum"]
