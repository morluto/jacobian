"""Hypergraph vertex containment probability kernel."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.probability.hypergraph_containment._models import (
    HypergraphVertexContainmentResult,
)

__all__ = ["compute_hypergraph_vertex_containment"]


def compute_hypergraph_vertex_containment(
    hypergraph: FiniteHypergraph,
    retention_probability: CanonicalRational,
) -> HypergraphVertexContainmentResult:
    """Return the complete vertex-containment profile of a hypergraph.

    For each k-subset of vertices, check if it contains a declared hyperedge.
    Count by k and compute the exact probability under independent vertex
    retention.
    """
    vertices = list(hypergraph.vertices)
    n = len(vertices)
    edges = [frozenset(members) for _, members in hypergraph.edges]

    counts = [0] * (n + 1)
    for mask in range(1 << n):
        selected = frozenset(vertices[i] for i in range(n) if mask & (1 << i))
        k = len(selected)
        contains_edge = any(edge <= selected for edge in edges)
        if contains_edge:
            counts[k] += 1

    total = 1 << n
    success = sum(counts)
    p = retention_probability.as_fraction()
    q = 1 - p
    prob = Fraction(0)
    for k in range(n + 1):
        prob += counts[k] * (p**k) * (q ** (n - k))

    return HypergraphVertexContainmentResult(
        hypergraph=hypergraph,
        retention_probability=retention_probability,
        containing_subset_counts=tuple(counts),
        total_state_count=total,
        success_count=success,
        probability=CanonicalRational.from_fraction(prob),
    )
