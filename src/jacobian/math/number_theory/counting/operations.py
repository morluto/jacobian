"""Exact arithmetic counting operations."""

from math import gcd

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.counting._models import (
    _MAX_BOX_COORD_DIGITS,
    _MAX_BOX_LINEAR_COEFFICIENT,
    _MAX_BOX_MODULUS,
    _MAX_FLOOR_SUM_N,
    _MAX_FLOOR_SUM_PARAM,
    CongruenceBoxCountResult,
    FloorSumResult,
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

    coordinates = (
        ("x_lo", x_lo),
        ("x_hi", x_hi),
        ("y_lo", y_lo),
        ("y_hi", y_hi),
    )
    for name, value in coordinates:
        if type(value) is not int:
            _reject((name,), f"{name}_type", f"{name} must be an exact integer")
        if abs(value) >= 10**_MAX_BOX_COORD_DIGITS:
            _reject(
                (name,),
                f"{name}_out_of_range",
                f"{name} exceeds the {_MAX_BOX_COORD_DIGITS}-digit bound",
            )
    if x_lo > x_hi:
        _reject(("x_lo", "x_hi"), "x_interval_invalid", "x_lo must be <= x_hi")
    if y_lo > y_hi:
        _reject(("y_lo", "y_hi"), "y_interval_invalid", "y_lo must be <= y_hi")
    if type(modulus) is not int or not 1 <= modulus <= _MAX_BOX_MODULUS:
        _reject(
            ("modulus",),
            "modulus_out_of_range",
            "modulus is outside the admitted range",
        )
    for name, value in (("u", u), ("v", v), ("c", c)):
        if type(value) is not int or not (
            -_MAX_BOX_LINEAR_COEFFICIENT <= value <= _MAX_BOX_LINEAR_COEFFICIENT
        ):
            _reject(
                (name,),
                f"{name}_out_of_range",
                f"{name} is outside the admitted range",
            )

    x_length = x_hi - x_lo + 1
    y_length = y_hi - y_lo + 1
    if y_length < x_length:
        x_lo, x_hi, y_lo, y_hi = y_lo, y_hi, x_lo, x_hi
        x_length = y_length
        u, v = v, u

    divisor = gcd(v, modulus)
    reduced_modulus = modulus // divisor
    inverse = 0 if reduced_modulus == 1 else pow(v // divisor, -1, reduced_modulus)

    count = 0
    # A short axis exposes fewer than one full period, so visiting its actual
    # representatives avoids turning a one-column box into a modulus-sized
    # scan.  Otherwise one representative per residue captures all distinct
    # congruence behavior and the multiplicity formula accounts for repeats.
    residues = range(x_lo, x_hi + 1) if x_length < modulus else range(modulus)
    for representative in residues:
        residue = representative % modulus
        x_count = (x_hi - residue) // modulus - (x_lo - 1 - residue) // modulus
        if not x_count:
            continue
        right_hand_side = c - u * residue
        if right_hand_side % divisor:
            continue
        y_residue = (right_hand_side // divisor * inverse) % reduced_modulus
        y_count = (y_hi - y_residue) // reduced_modulus
        y_count -= (y_lo - 1 - y_residue) // reduced_modulus
        count += x_count * y_count
    return count


__all__ = [
    "congruence_box_count",
    "floor_sum",
    "verify_congruence_box_count",
    "verify_floor_sum",
]


def verify_floor_sum(claim: FloorSumResult) -> bool:
    """Check a serialized floor-sum claim against its retained parameters."""
    try:
        expected = floor_sum(claim.n, claim.m, claim.a, claim.b)
        actual = claim.value
    except (OperationDomainValidationError, TypeError, ValueError):
        return False
    return actual == expected


def verify_congruence_box_count(claim: CongruenceBoxCountResult) -> bool:
    """Check a serialized lattice-point count against its retained box and equation."""
    try:
        expected = congruence_box_count(
            x_lo=claim.x_lo,
            x_hi=claim.x_hi,
            y_lo=claim.y_lo,
            y_hi=claim.y_hi,
            u=claim.u,
            v=claim.v,
            c=claim.c,
            modulus=claim.modulus,
        )
    except (OperationDomainValidationError, TypeError, ValueError):
        return False
    return claim.count == expected
