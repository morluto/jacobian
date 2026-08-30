"""Canonical uniform-subset intersection graph constructor."""

from __future__ import annotations

from itertools import combinations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.uniform_subset_intersection._models import (
    IntersectionRelation,
    UniformSubsetIntersectionResult,
    _admit_uniform_subset_intersection,
    _UniformSubsetIntersectionPlan,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["construct_uniform_subset_intersection_graph"]


def construct_uniform_subset_intersection_graph(
    ground_set_size: int,
    subset_cardinality: int,
    threshold: int,
    relation: IntersectionRelation,
) -> UniformSubsetIntersectionResult:
    """Construct a graph on k-subsets of [n] with a threshold relation.

    Vertices are all k-subsets of {0,...,n-1}, labelled by the sorted
    comma-separated subset. An edge joins two subsets when their
    intersection size satisfies the declared relation with the threshold.
    """
    try:
        plan = _admit_uniform_subset_intersection(
            ground_set_size, subset_cardinality, threshold, relation
        )
    except (TypeError, ValueError) as error:
        raise OperationDomainValidationError(
            location=(),
            code="graph.uniform_subset_intersection.request_not_admitted",
            message=str(error),
        ) from error
    return _construct_uniform_subset_intersection_graph_from_plan(
        ground_set_size, subset_cardinality, threshold, relation, plan
    )


def _construct_uniform_subset_intersection_graph_from_plan(
    ground_set_size: int,
    subset_cardinality: int,
    threshold: int,
    relation: IntersectionRelation,
    plan: _UniformSubsetIntersectionPlan,
) -> UniformSubsetIntersectionResult:
    """Construct a graph from one already-computed request admission."""

    subsets = (
        ((),)
        if subset_cardinality == 0
        else tuple(combinations(range(ground_set_size), subset_cardinality))
    )
    assert len(subsets) == plan.vertex_count

    if len(subsets) == 1:
        graph = SimpleUndirectedGraph(
            vertices=(_subset_label(subsets[0]),),
            edges=(),
        )
        return UniformSubsetIntersectionResult(
            ground_set_size=ground_set_size,
            subset_cardinality=subset_cardinality,
            threshold=threshold,
            relation=relation,
            graph=graph,
        )

    vertices_labels = [_subset_label(s) for s in subsets]
    edges: list[tuple[str, str]] = []

    for i in range(len(subsets)):
        si = set(subsets[i])
        for j in range(i + 1, len(subsets)):
            intersection_size = len(si.intersection(subsets[j]))
            if relation == "INTERSECTION_LT_THRESHOLD":
                adjacent = intersection_size < threshold
            else:
                adjacent = intersection_size == threshold
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
    assert len(graph.edges) == plan.edge_count
    return UniformSubsetIntersectionResult(
        ground_set_size=ground_set_size,
        subset_cardinality=subset_cardinality,
        threshold=threshold,
        relation=relation,
        graph=graph,
    )


def _subset_label(subset: tuple[int, ...]) -> str:
    """Canonical label for a k-subset."""
    return "{" + ",".join(str(x) for x in subset) + "}"
