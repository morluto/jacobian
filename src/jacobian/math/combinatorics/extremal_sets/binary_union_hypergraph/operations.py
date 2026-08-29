"""Binary-union relation hypergraph kernel."""

from __future__ import annotations

from jacobian.math.combinatorics.extremal_sets.binary_union_hypergraph._models import (
    BinaryUnionHypergraphResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

__all__ = ["compute_binary_union_hypergraph"]


def compute_binary_union_hypergraph(
    sets: tuple[tuple[int, ...], ...],
) -> BinaryUnionHypergraphResult:
    """Construct the 3-uniform hypergraph of A ∪ B = C relations.

    For every triple (i,j,k) of distinct indices with S_i ∪ S_j = S_k,
    add one hyperedge.
    """
    n = len(sets)
    vertices = tuple(str(i) for i in range(n))
    edges: list[tuple[str, tuple[str, ...]]] = []

    edge_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            union = set(sets[i]) | set(sets[j])
            for k in range(n):
                if k == i or k == j:
                    continue
                if set(sets[k]) == union:
                    edges.append(
        (f"rel_{edge_idx}", (str(i), str(j), str(k)))
                    )
                    edge_idx += 1

    hypergraph = FiniteHypergraph(
        vertices=vertices,
        edges=tuple(edges),
    )

    return BinaryUnionHypergraphResult(
        sets=sets,
        hypergraph=hypergraph,
        relation_count=len(edges),
    )
