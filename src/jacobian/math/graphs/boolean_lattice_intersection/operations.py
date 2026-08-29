"""Boolean-lattice intersection graph constructor."""

from __future__ import annotations

from jacobian.math.graphs.boolean_lattice_intersection._models import (
    BooleanLatticeIntersectionResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["construct_boolean_lattice_intersection_graph"]


def construct_boolean_lattice_intersection_graph(
    ground_set_size: int,
    threshold: int,
    relation: str,
) -> BooleanLatticeIntersectionResult:
    """Construct a graph on the Boolean lattice 2^[n].

    Vertices are all subsets of [n], labelled by canonical subset strings.
    An edge joins two distinct subsets when their intersection size
    satisfies the declared relation with the threshold.
    """
    n = ground_set_size
    all_subsets = [_subset_label(s) for s in _all_subsets(n)]

    edges: list[tuple[str, str]] = []
    for i in range(len(all_subsets)):
        si = _subset_from_label(all_subsets[i])
        for j in range(i + 1, len(all_subsets)):
            sj = _subset_from_label(all_subsets[j])
            intersection_size = len(si & sj)
            if relation == "INTERSECTION_EQ":
                adjacent = intersection_size == threshold
            elif relation == "INTERSECTION_LT":
                adjacent = intersection_size < threshold
            else:
                adjacent = intersection_size > threshold
            if adjacent:
                a, b = all_subsets[i], all_subsets[j]
                if a < b:
                    edges.append((a, b))
                else:
                    edges.append((b, a))

    graph = SimpleUndirectedGraph(
        vertices=tuple(all_subsets),
        edges=tuple(edges),
    )
    return BooleanLatticeIntersectionResult(
        ground_set_size=n,
        threshold=threshold,
        relation=relation,
        graph=graph,
    )


def _all_subsets(n: int) -> list[frozenset[int]]:
    if n == 0:
        return [frozenset()]
    result = []
    for mask in range(1 << n):
        result.append(frozenset(i for i in range(n) if mask & (1 << i)))
    return result


def _subset_label(subset: frozenset[int]) -> str:
    if not subset:
        return "{}"
    return "{" + ",".join(str(i) for i in sorted(subset)) + "}"


def _subset_from_label(label: str) -> frozenset[int]:
    if label == "{}":
        return frozenset()
    inner = label[1:-1]
    return frozenset(int(x) for x in inner.split(","))
