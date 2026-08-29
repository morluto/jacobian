"""Boolean-lattice intersection graph kernel."""

from __future__ import annotations

from itertools import combinations

from jacobian.math.graphs.boolean_lattice_intersection._models import (
    BooleanLatticeIntersectionResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["construct_boolean_lattice_intersection_graph"]


def construct_boolean_lattice_intersection_graph(
    n: int,
    intersection_cardinality: int,
    relation: str,
) -> BooleanLatticeIntersectionResult:
    """Construct a graph whose vertices are all subsets of [n].

    An edge joins two distinct subsets when their intersection satisfies
    the declared relation with the given cardinality.
    """
    subsets = []
    for size in range(n + 1):
        for combo in combinations(range(n), size):
            subsets.append(combo)

    labels = [_label(s) for s in subsets]

    edges: list[tuple[str, str]] = []
    for i in range(len(subsets)):
        for j in range(i + 1, len(subsets)):
            intersection_size = len(set(subsets[i]) & set(subsets[j]))
            if _check_relation(intersection_size, intersection_cardinality, relation):
                edges.append((labels[i], labels[j]))

    edges.sort()
    graph = SimpleUndirectedGraph(
        vertices=tuple(labels),
        edges=tuple(edges),
    )

    return BooleanLatticeIntersectionResult(
        n=n,
        intersection_cardinality=intersection_cardinality,
        relation=relation,
        graph=graph,
    )


def _label(subset: tuple[int, ...]) -> str:
    if not subset:
        return "L0_"
    return f"L{len(subset)}_" + "_".join(str(x) for x in subset)


def _check_relation(intersection_size: int, threshold: int, relation: str) -> bool:
    checks = {
        "INTERSECTION_EQ_THRESHOLD": intersection_size == threshold,
        "INTERSECTION_LT_THRESHOLD": intersection_size < threshold,
        "INTERSECTION_GT_THRESHOLD": intersection_size > threshold,
    }
    return checks.get(relation, False)
