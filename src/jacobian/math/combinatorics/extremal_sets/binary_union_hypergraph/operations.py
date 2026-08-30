"""Binary-union relation hypergraph kernel."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.extremal_sets.binary_union_hypergraph._models import (
    BinaryUnionHypergraphResult,
    _binary_union_admission_error,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

__all__ = ["compute_binary_union_hypergraph"]


def compute_binary_union_hypergraph(
    sets: tuple[tuple[int, ...], ...],
) -> BinaryUnionHypergraphResult:
    """Construct the 3-uniform hypergraph of A union B = C relations.

    For every triple (i,j,k) of distinct indices with S_i union S_j = S_k,
    add one hyperedge.
    """
    failure = _binary_union_admission_error(sets)
    if failure is not None:
        code, message = failure
        raise OperationDomainValidationError(
            location=("sets",), code=f"binary_union.{code}", message=message
        )
    n = len(sets)
    vertices = tuple(str(i) for i in range(n))
    edges: list[tuple[str, tuple[str, ...]]] = []
    set_index = {frozenset(values): index for index, values in enumerate(sets)}

    edge_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            union = set(sets[i]) | set(sets[j])
            k = set_index.get(frozenset(union))
            if k is not None and k not in (i, j):
                members = tuple(sorted((str(i), str(j), str(k))))
                edges.append((f"rel_{edge_idx}", members))
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
