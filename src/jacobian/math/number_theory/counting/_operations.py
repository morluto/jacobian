"""Domain-owned arithmetic counting operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.counting._models import (
    _MAX_BOX_AREA,
    CongruenceBoxCountRequest,
    CongruenceBoxCountResult,
    FloorSumRequest,
    FloorSumResult,
)


def _floor_sum_euclidean(n: int, m: int, a: int, b: int) -> int:
    """Exact ``sum_{i=0}^{n-1} floor((a*i + b) / m)`` by Euclidean-like halving.

    Standard identity (AtCoder Library ``floor_sum``): reduce a >= m and
    b >= m with closed-form quotient sums, then recurse on the shrunk pair
    ``(m mod a-style swap)``.  Each iteration at least halves the larger of
    ``(a, m)``, so the work is O(log m * log(a/m))-ish — logarithmic in the
    parameters and independent of ``n``.
    """

    answer = 0
    while True:
        if a >= m:
            answer += (n - 1) * n // 2 * (a // m)
            a %= m
        if b >= m:
            answer += n * (b // m)
            b %= m
        y_max = a * n + b
        if y_max < m:
            return answer
        n = y_max // m
        b = y_max % m
        m, a = a, m


def compute_floor_sum(request: FloorSumRequest) -> FloorSumResult:
    """Compute sum_{i=0}^{n-1} floor((a*i + b) / m) exactly."""
    total = _floor_sum_euclidean(request.n, request.m, request.a, request.b)
    return FloorSumResult(value=format_canonical_integer(total))


def compute_congruence_box_count(
    request: CongruenceBoxCountRequest,
) -> CongruenceBoxCountResult:
    """Count lattice points in a box satisfying u*x + v*y = c (mod modulus)."""
    area = (request.x_hi - request.x_lo + 1) * (request.y_hi - request.y_lo + 1)
    if area > _MAX_BOX_AREA:
        raise OperationDomainValidationError(
            location=("x_lo", "x_hi", "y_lo", "y_hi"),
            code="arithmetic_counting.box_area_exceeds_budget",
            message="box area exceeds the computational budget",
        )
    modulus = request.modulus
    count = 0
    for x in range(request.x_lo, request.x_hi + 1):
        for y in range(request.y_lo, request.y_hi + 1):
            if (request.u * x + request.v * y - request.c) % modulus == 0:
                count += 1
    return CongruenceBoxCountResult(count=count, modulus=modulus)


__all__ = ["compute_congruence_box_count", "compute_floor_sum"]
