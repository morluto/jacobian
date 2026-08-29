"""Binary-union relation hypergraph constructor."""

from __future__ import annotations

from jacobian.math.combinatorics.extremal_sets._models import (
    BinaryUnionRelationResult,
    UnionRelationRow,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

__all__ = ["construct_binary_union_relation"]


def construct_binary_union_relation(
    family: tuple[tuple[int, ...], ...],
) -> BinaryUnionRelationResult:
    """Compute the complete binary-union relation of a set family.

    For every pair of distinct source indices (i, j) with i < j, compute
    S_i union S_j and check whether the result equals any source member S_k
    with k distinct from both i and j. If so, record the triple (i, j, k).
    """
    m = len(family)
    sets = [frozenset(s) for s in family]
    union_to_index: dict[frozenset[int], int] = {}
    for i, s in enumerate(sets):
        union_to_index.setdefault(s, i)

    rows: list[UnionRelationRow] = []
    for i in range(m):
        for j in range(i + 1, m):
            union = sets[i] | sets[j]
            k = union_to_index.get(union)
            if k is not None and k != i and k != j:
                rows.append(UnionRelationRow(operand_i=i, operand_j=j, result_k=k))

    vertices = tuple(str(i) for i in range(m))
    hyper_edges: list[tuple[str, tuple[str, ...]]] = []
    for idx, row in enumerate(rows):
        edge_id = f"edge_{idx}"
        members = tuple(
            sorted([str(row.operand_i), str(row.operand_j), str(row.result_k)])
        )
        hyper_edges.append((edge_id, members))

    hypergraph = FiniteHypergraph(
        vertices=vertices,
        edges=tuple(hyper_edges),
    )
    return BinaryUnionRelationResult(
        family=family,
        rows=tuple(rows),
        hypergraph=hypergraph,
    )
