"""Exact kernel for divisibility edge profiles with quotient and LPF."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from jacobian._execution import (
    OperationExecutionTimeoutError,
    current_request_execution,
)
from jacobian.math.number_theory._factorization_kernels import (
    BoundedFactorizationFailure,
    _bounded_direct_factorization,
)


class FactorizationIncompleteError(RuntimeError):
    """The killable factorization worker did not establish a complete result."""

    def __init__(self, failure: BoundedFactorizationFailure | None) -> None:
        self.failure = failure
        super().__init__("bounded factorization did not establish a complete result")


@dataclass(frozen=True, slots=True)
class DivisibilityEdgeData:
    """Private canonical data for one edge."""

    source: str
    target: str
    quotient: int
    least_prime_factor: int


def _least_prime_factor(n: int) -> int:
    """Return the least prime factor of n > 1."""
    if n <= 1:
        raise ValueError(f"least_prime_factor requires n > 1, got {n}")
    failures: list[BoundedFactorizationFailure] = []
    execution = current_request_execution()
    timeout_seconds = 60.0
    if execution is not None and execution.deadline is not None:
        timeout_seconds = execution.deadline - monotonic()
        if timeout_seconds <= 0:
            raise OperationExecutionTimeoutError(
                "divisibility edge factorization request deadline expired"
            )
    factors = _bounded_direct_factorization(
        n, timeout_seconds=timeout_seconds, failure=failures
    )
    if factors is None:
        raise FactorizationIncompleteError(failures[0] if failures else None)
    return min(int(factor.prime) for factor in factors)


def construct_divisibility_edge_profile(
    values: tuple[str, ...],
    edge_plan: tuple[tuple[int, int, int], ...],
) -> list[DivisibilityEdgeData]:
    """Return the complete directed divisibility edge table.

    For each pair (a, b) with a != b where a divides b, record:
    - source = a, target = b
    - quotient = b / a
    - least_prime_factor = least prime factor of the quotient
    """
    edges: list[DivisibilityEdgeData] = []
    lpf_cache: dict[int, int] = {}

    for i, j, quotient in edge_plan:
        lpf = lpf_cache.get(quotient)
        if lpf is None:
            lpf = _least_prime_factor(quotient)
            lpf_cache[quotient] = lpf
        edges.append(
            DivisibilityEdgeData(
                source=values[i],
                target=values[j],
                quotient=quotient,
                least_prime_factor=lpf,
            )
        )

    return edges
