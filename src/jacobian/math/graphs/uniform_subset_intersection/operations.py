"""Canonical uniform-subset intersection graph constructor."""

from __future__ import annotations

from itertools import combinations

from jacobian.math.graphs.uniform_subset_intersection._models import (
    UniformSubsetIntersectionRequest,
    UniformSubsetIntersectionResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["construct_uniform_subset_intersection_graph"]


def construct_uniform_subset_intersection_graph(
    request: UniformSubsetIntersectionRequest,
) -> UniformSubsetIntersectionResult:
    """Construct a graph on k-subsets of [n] with a threshold relation.

    Vertices are all k-subsets of {0,...,n-1}, labelled by the sorted
    comma-separated subset. An edge joins two subsets when their
    intersection size satisfies the declared relation with the threshold.
    """
    n = request.ground_set_size
    k = request.subset_cardinality
    t = request.threshold
    relation = request.relation

    if k == 0 or k > n:
        graph = SimpleUndirectedGraph(vertices=(), edges=())
        return UniformSubsetIntersectionResult(
            ground_set_size=n,
            subset_cardinality=k,
            threshold=t,
            relation=relation,
            graph=graph,
        )

    ground = list(range(n))
    subsets = list(combinations(ground, k))

    if len(subsets) == 1:
        graph = SimpleUndirectedGraph(
            vertices=(_subset_label(subsets[0]),),
            edges=(),
        )
        return UniformSubsetIntersectionResult(
            ground_set_size=n,
            subset_cardinality=k,
            threshold=t,
            relation=relation,
            graph=graph,
        )

    vertices_labels = [_subset_label(s) for s in subsets]
    edges: list[tuple[str, str]] = []

    for i in range(len(subsets)):
        si = set(subsets[i])
        for j in range(i + 1, len(subsets)):
            intersection_size = len(si & set(subsets[j]))
            if relation == "INTERSECTION_LT_THRESHOLD":
                adjacent = intersection_size < t
            else:
                adjacent = intersection_size == t
            if adjacent:
                a, b = vertices_labels[i], vertices_labels[j]
                if a < b:
                    edges.append((a, b))
                else:
                    edges.append((b, a))

    graph = SimpleUndirectedGraph(
        vertices=tuple(vertices_labels),
        edges=tuple(edges),
    )
    return UniformSubsetIntersectionResult(
        ground_set_size=n,
        subset_cardinality=k,
        threshold=t,
        relation=relation,
        graph=graph,
    )


def _subset_label(subset: tuple[int, ...]) -> str:
    """Canonical label for a k-subset."""
    return "{" + ",".join(str(x) for x in subset) + "}"
