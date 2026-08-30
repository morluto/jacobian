"""Divisibility-sum triples hypergraph kernel."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.divisibility_sum_triples._models import (
    MAX_DIVISIBILITY_SUM_INTERVAL_SIZE,
    DivisibilitySumTriplesResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

__all__ = ["construct_divisibility_sum_triples"]


def construct_divisibility_sum_triples(
    lower: int,
    upper: int,
) -> DivisibilitySumTriplesResult:
    """Construct the 3-uniform hypergraph of divisibility-sum triples.

    On an integer interval [L,U], for every triple (a,b,c) with a < b < c
    and a | (b+c), create one hyperedge.
    """
    if lower < 1 or upper < lower:
        raise OperationDomainValidationError(
            location=("lower", "upper"),
            code="divisibility_sum_triples.positive_ordered_interval",
            message="divisibility-sum triples require 1 <= lower <= upper",
        )
    if upper - lower + 1 > MAX_DIVISIBILITY_SUM_INTERVAL_SIZE:
        raise OperationDomainValidationError(
            location=("lower", "upper"),
            code="divisibility_sum_triples.interval_size",
            message="divisibility-sum triples admit at most 42 interval values",
        )
    interval_size = upper - lower + 1
    if (
        len(str(upper)) > 64
        or len(
            f"triple_{interval_size * (interval_size - 1) * (interval_size - 2) // 6}"
        )
        > 64
    ):
        raise OperationDomainValidationError(
            location=("lower", "upper"),
            code="divisibility_sum_triples.label_size",
            message="divisibility-sum vertex and edge labels exceed the 64-byte carrier",
        )
    vertices = tuple(str(i) for i in range(lower, upper + 1))
    edges: list[tuple[str, tuple[str, ...]]] = []

    edge_idx = 0
    for a in range(lower, upper + 1):
        for b in range(a + 1, upper + 1):
            for c in range(b + 1, upper + 1):
                if (b + c) % a == 0:
                    edges.append((f"triple_{edge_idx}", (str(a), str(b), str(c))))
                    edge_idx += 1

    hypergraph = FiniteHypergraph(
        vertices=vertices,
        edges=tuple(edges),
    )

    return DivisibilitySumTriplesResult(
        lower=lower,
        upper=upper,
        hypergraph=hypergraph,
    )
