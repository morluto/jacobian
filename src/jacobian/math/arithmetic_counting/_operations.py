"""Domain-owned arithmetic counting operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.arithmetic_counting._models import (
    CongruenceBoxCountRequest,
    CongruenceBoxCountResult,
    FloorSumRequest,
    FloorSumResult,
)


def compute_floor_sum(request: FloorSumRequest) -> FloorSumResult:
    """Compute sum_{i=0}^{n-1} floor((a*i + b) / m) exactly."""
    total = 0
    for index in range(request.n):
        total += (request.a * index + request.b) // request.m
    return FloorSumResult(value=format_canonical_integer(total))


def compute_congruence_box_count(
    request: CongruenceBoxCountRequest,
) -> CongruenceBoxCountResult:
    """Count lattice points in a box satisfying u*x + v*y = c (mod modulus)."""
    modulus = request.modulus
    count = 0
    for x in range(request.x_lo, request.x_hi + 1):
        for y in range(request.y_lo, request.y_hi + 1):
            if (request.u * x + request.v * y - request.c) % modulus == 0:
                count += 1
    return CongruenceBoxCountResult(count=count, modulus=modulus)


__all__ = ["compute_congruence_box_count", "compute_floor_sum"]
