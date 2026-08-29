"""Divisibility-sum triple hypergraph constructor."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.divisibility_sum_triples._models import (
    DivisibilitySumTriplesResult,
    _validate_interval,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

__all__ = ["construct_divisibility_sum_triples_hypergraph"]


def construct_divisibility_sum_triples_hypergraph(
    lower_bound: int,
    upper_bound: int,
) -> DivisibilitySumTriplesResult:
    """Construct the 3-uniform hypergraph of divisibility-sum triples.

    Vertices are the integers in [L, U]. Edges are the increasing triples
    (a, b, c) with L <= a < b < c <= U and a | (b + c).
    """
    try:
        _validate_interval(lower_bound, upper_bound)
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=(), code=error.type, message=str(error)
        ) from error

    vertices = tuple(str(i) for i in range(lower_bound, upper_bound + 1))
    edges: list[tuple[str, tuple[str, ...]]] = []
    edge_index = 0

    for a in range(lower_bound, upper_bound + 1):
        for b in range(a + 1, upper_bound + 1):
            for c in range(b + 1, upper_bound + 1):
                if a != 0 and (b + c) % a == 0:
                    edges.append((f"edge_{edge_index}", (str(a), str(b), str(c))))
                    edge_index += 1

    hypergraph = FiniteHypergraph(
        vertices=vertices,
        edges=tuple(edges),
    )
    return DivisibilitySumTriplesResult(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        hypergraph=hypergraph,
    )
