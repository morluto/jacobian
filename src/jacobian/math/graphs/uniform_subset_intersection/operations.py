"""Uniform-subset intersection graph kernel."""

from __future__ import annotations

from itertools import combinations

from jacobian.math.graphs.uniform_subset_intersection._models import (
    UniformSubsetIntersectionResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["construct_uniform_subset_intersection_graph"]


def construct_uniform_subset_intersection_graph(
    n: int,
    k: int,
    threshold: int,
    relation: str,
) -> UniformSubsetIntersectionResult:
    """Construct a graph whose vertices are k-subsets of [n].

    An edge joins two k-subsets when their intersection satisfies the
    declared relation with the threshold.
    """
    subsets = list(combinations(range(n), k))
    # Use sorted tuple string as vertex label
    labels = [f"L{len(s)}_" + "_".join(str(x) for x in s) for s in subsets]

    edges: list[tuple[str, str]] = []
    for i in range(len(subsets)):
        for j in range(i + 1, len(subsets)):
            intersection_size = len(set(subsets[i]) & set(subsets[j]))
            if _check_relation(intersection_size, threshold, relation):
                edges.append((labels[i], labels[j]))

    edges.sort()
    graph = SimpleUndirectedGraph(
        vertices=tuple(labels),
        edges=tuple(edges),
    )

    return UniformSubsetIntersectionResult(
        n=n,
        k=k,
        threshold=threshold,
        relation=relation,
        graph=graph,
    )


def _check_relation(intersection_size: int, threshold: int, relation: str) -> bool:
    if relation == "INTERSECTION_LT_THRESHOLD":
        return intersection_size < threshold
    elif relation == "INTERSECTION_EQ_THRESHOLD":
        return intersection_size == threshold
    elif relation == "INTERSECTION_GT_THRESHOLD":
        return intersection_size > threshold
    return False
