"""Exact kernel for divisibility edge profiles with quotient and LPF."""

from __future__ import annotations

from dataclasses import dataclass

from sympy.ntheory.factor_ import factorint


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
    factors = factorint(n)
    return min(factors.keys())


def construct_divisibility_edge_profile(
    values: tuple[str, ...],
) -> list[DivisibilityEdgeData]:
    """Return the complete directed divisibility edge table.

    For each pair (a, b) with a != b where a divides b, record:
    - source = a, target = b
    - quotient = b / a
    - least_prime_factor = least prime factor of the quotient
    """
    int_values = [int(v) for v in values]
    n = len(int_values)
    edges: list[DivisibilityEdgeData] = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if int_values[j] % int_values[i] == 0:
                quotient = int_values[j] // int_values[i]
                if quotient <= 1:
                    continue
                lpf = _least_prime_factor(quotient)
                edges.append(
                    DivisibilityEdgeData(
                        source=values[i],
                        target=values[j],
                        quotient=quotient,
                        least_prime_factor=lpf,
                    )
                )

    return edges
