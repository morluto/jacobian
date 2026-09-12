"""Typed contracts for 3-term progression hypergraph construction."""

from __future__ import annotations

from math import gcd, prod

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_VERTICES,
    FiniteHypergraph,
)
from jacobian.math.groups.finite_abelian import (
    FiniteAbelianGroupElement,
    FiniteAbelianProductGroup,
)


def _torsion_cardinality(group: FiniteAbelianProductGroup, order: int) -> int:
    """Return the cardinality of the subgroup killed by ``order``.

    In one cyclic factor ``Z/mZ``, precisely ``gcd(order, m)`` elements solve
    ``order * x = 0``.  The product formula keeps this pre-enumeration count
    exact without materializing any group elements.
    """

    return prod(gcd(order, modulus) for modulus in group.moduli)


def progression_edge_bound(group: FiniteAbelianProductGroup) -> int:
    """Return the exact number of nondegenerate unordered 3-AP edges.

    An ordered pair ``(a, d)`` yields a nondegenerate progression exactly when
    ``2d != 0``, giving ``|G| (|G| - |G[2]|)`` encodings.  Such an edge has two
    encodings from reversal, except when ``3d = 0``: its nonzero order-three
    difference gives all three vertices as possible centers and therefore six
    encodings.  There are ``|G| (|G[3]| - 1)`` encodings in that exceptional
    class, so subtracting its two-encoding contribution and restoring its
    six-encoding contribution gives the formula below.
    """

    group_order = group.order
    two_torsion = _torsion_cardinality(group, 2)
    three_torsion = _torsion_cardinality(group, 3)
    ordinary_encodings = group_order * (group_order - two_torsion)
    order_three_encodings = group_order * (three_torsion - 1)
    return (
        ordinary_encodings - order_three_encodings
    ) // 2 + order_three_encodings // 6


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
