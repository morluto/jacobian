"""Typed contracts for 3-term progression hypergraph construction."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    MAX_VERTICES,
    FiniteHypergraph,
)


def _progression_edge_count(group_order: int) -> int:
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


def _fits_hypergraph_representation(group_order: int) -> bool:
    edge_count = _progression_edge_count(group_order)
    return edge_count <= MAX_EDGES and 3 * edge_count <= MAX_TOTAL_INCIDENCES


MAX_GROUP_ORDER: int = max(
    group_order
    for group_order in range(2, MAX_VERTICES + 1)
    if _fits_hypergraph_representation(group_order)
)


class ProgressionHypergraphRequest(StrictModel):
    """Order of the cyclic group Z/nZ for 3-AP hypergraph construction."""

    group_order: int = Field(ge=2, le=MAX_GROUP_ORDER)


class ProgressionHypergraphResult(StrictModel):
    """The 3-uniform hypergraph of all 3-term arithmetic progressions in Z/nZ."""

    group_order: int
    hypergraph: FiniteHypergraph


__all__ = [
    "MAX_GROUP_ORDER",
    "ProgressionHypergraphRequest",
    "ProgressionHypergraphResult",
]
