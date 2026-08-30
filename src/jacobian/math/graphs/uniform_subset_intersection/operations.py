"""Uniform-subset intersection graph kernel."""

from __future__ import annotations

from itertools import combinations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.uniform_subset_intersection._models import (
    UniformSubsetIntersectionRelation,
    UniformSubsetIntersectionResult,
    _uniform_subset_admission_error,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["construct_uniform_subset_intersection_graph"]


def construct_uniform_subset_intersection_graph(
    n: int,
    k: int,
    threshold: int,
    relation: UniformSubsetIntersectionRelation,
) -> UniformSubsetIntersectionResult:
    """Construct a graph whose vertices are k-subsets of [n].

    An edge joins two k-subsets when their intersection satisfies the
    declared relation with the threshold.
    """
    if relation not in {
        "INTERSECTION_LT_THRESHOLD",
        "INTERSECTION_EQ_THRESHOLD",
        "INTERSECTION_GT_THRESHOLD",
    }:
        raise OperationDomainValidationError(
            location=("relation",),
            code="uniform_subset.invalid_relation",
            message="relation must be one of the declared intersection relations",
        )
    failure = _uniform_subset_admission_error(n, k)
    if failure is not None:
        code, message = failure
        raise OperationDomainValidationError(
            location=("n", "k"),
            code=f"uniform_subset.{code}",
            message=message,
        )
    subsets = list(combinations(range(n), k))
    # Use sorted tuple string as vertex label
    labels = [f"L{len(s)}_" + "_".join(str(x) for x in s) for s in subsets]

    edges: list[tuple[str, str]] = []
    for i in range(len(subsets)):
        for j in range(i + 1, len(subsets)):
            intersection_size = len(set(subsets[i]) & set(subsets[j]))
            if _check_relation(intersection_size, threshold, relation):
                left, right = sorted((labels[i], labels[j]))
                edges.append((left, right))

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
    checks = {
        "INTERSECTION_LT_THRESHOLD": intersection_size < threshold,
        "INTERSECTION_EQ_THRESHOLD": intersection_size == threshold,
        "INTERSECTION_GT_THRESHOLD": intersection_size > threshold,
    }
    return checks.get(relation, False)
