"""Divisibility edge profile with quotient and LPF declaration."""

from __future__ import annotations

from jacobian.catalog._examples import example
from jacobian.math.number_theory._divisibility_edge_profile_kernels import (
    construct_divisibility_edge_profile,
)
from jacobian.math.number_theory._divisibility_edge_profile_models import (
    DivisibilityEdge,
    DivisibilityEdgeProfileRequest,
    DivisibilityEdgeProfileResult,
)
from jacobian.math.number_theory._support import number_theory_operation


def compute_divisibility_edge_profile(
    request: DivisibilityEdgeProfileRequest,
) -> DivisibilityEdgeProfileResult:
    """Return the complete directed divisibility edge table with quotient and LPF."""
    data = construct_divisibility_edge_profile(request.values)
    edges = tuple(
        DivisibilityEdge(
            source=d.source,
            target=d.target,
            quotient=d.quotient,
            least_prime_factor=d.least_prime_factor,
        )
        for d in data
    )
    return DivisibilityEdgeProfileResult(values=request.values, edges=edges)


DIVISIBILITY_EDGE_PROFILE_OPERATION = number_theory_operation(
    "integer.divisibility_edge_profile.compute",
    "Profile quotient and least-prime-factor data on divisibility edges",
    "Given an ordered finite source set of positive integers, return the "
    "complete directed proper-divisibility edge table. Each edge a -> b "
    "carries the quotient b/a and its least prime factor.",
    DivisibilityEdgeProfileRequest,
    DivisibilityEdgeProfileResult,
    compute_divisibility_edge_profile,
    "number-theory",
    "divisibility",
    "primitive-set",
    "least-prime-factor",
    "exact",
    examples=(
        example(
            "divisibility_edges_24612",
            "For (2,4,6,12), profile all proper-divisibility edges with "
            "quotient and least-prime-factor data; values must be positive "
            "canonical decimal integers.",
            {"values": ["2", "4", "6", "12"]},
        ),
    ),
)


__all__ = [
    "DIVISIBILITY_EDGE_PROFILE_OPERATION",
    "compute_divisibility_edge_profile",
]
