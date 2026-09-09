"""Typed contracts for 3-term progression hypergraph construction."""

from __future__ import annotations

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_VERTICES,
    FiniteHypergraph,
)
from jacobian.math.groups.finite_abelian import (
    FiniteAbelianGroupElement,
    FiniteAbelianProductGroup,
)


def _cyclic_progression_edge_count(group_order: int) -> int:
    """Return the exact number of distinct 3-AP edges in ``Z/group_order Z``.

    The ordered construction has two encodings for an ordinary edge.  When
    three divides the group order, each coset of the order-three subgroup has
    six encodings, so those exceptional edges require a correction.
    """
    valid_differences = group_order - 2 if group_order % 2 == 0 else group_order - 1
    ordered_progressions = group_order * valid_differences
    edge_count = ordered_progressions // 2
    if group_order % 3 == 0:
        edge_count -= 2 * (group_order // 3)
    return edge_count


def progression_edge_bound(group: FiniteAbelianProductGroup) -> int:
    """Return a sound pre-enumeration edge bound for the supplied group."""

    if len(group.moduli) == 1:
        return _cyclic_progression_edge_count(group.order)
    return group.order * (group.order - 1) // 2


MAX_GROUP_ORDER = MAX_VERTICES


class ProgressionHypergraphRequest(StrictModel):
    """Finite Abelian product group whose three-point progressions are requested."""

    group: FiniteAbelianProductGroup


class ProgressionVertexBinding(StrictModel):
    """Binding from one hypergraph vertex to its canonical group element."""

    vertex: str
    element: FiniteAbelianGroupElement


class ProgressionHypergraphResult(StrictModel):
    """Complete 3-uniform progression hypergraph of a finite Abelian group."""

    group: FiniteAbelianProductGroup
    vertex_elements: tuple[ProgressionVertexBinding, ...]
    hypergraph: FiniteHypergraph


__all__ = [
    "MAX_GROUP_ORDER",
    "ProgressionHypergraphRequest",
    "ProgressionHypergraphResult",
    "ProgressionVertexBinding",
    "progression_edge_bound",
]
