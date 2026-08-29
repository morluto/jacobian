"""Exact non-coprimality conflict-graph construction."""

from __future__ import annotations

from itertools import combinations
from math import gcd as exact_gcd

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.number_theory.non_coprimality_graph._models import (
    MAX_NON_COPRIMALITY_GRAPH_VERTICES,
    NonCoprimalityGraphResult,
)


def non_coprimality_graph(
    vertices: tuple[str, ...],
) -> NonCoprimalityGraphResult:
    """Build the canonical simple non-coprimality conflict graph.

    The supplied ``vertices`` are canonical decimal integer strings. The
    returned :class:`SimpleUndirectedGraph` has one vertex per supplied integer
    — sorted by numeric value — and one edge per distinct pair with gcd
    greater than one.
    """

    n = len(vertices)

    if n > MAX_NON_COPRIMALITY_GRAPH_VERTICES:
        raise OperationDomainValidationError(
            location=("elements",),
            code="number_theory.non_coprimality_graph.too_many_elements",
            message=(
                "the non-coprimality graph admits at most "
                f"{MAX_NON_COPRIMALITY_GRAPH_VERTICES} integers"
            ),
        )

    if n == 0:
        return NonCoprimalityGraphResult(
            graph=SimpleUndirectedGraph(vertices=(), edges=())
        )

    values: list[int] = []
    for index, label in enumerate(vertices):
        value = int(label)
        if value <= 0:
            raise OperationDomainValidationError(
                location=("elements", index),
                code="number_theory.non_coprimality_graph.non_positive",
                message="all integers must be positive",
            )
        values.append(value)

    # Sort vertices by numeric value so the graph is canonical.
    order = sorted(range(n), key=lambda i: values[i])
    sorted_labels = tuple(vertices[i] for i in order)
    sorted_values = [values[i] for i in order]

    edges: list[tuple[str, str]] = []
    for i, j in combinations(range(n), 2):
        if exact_gcd(sorted_values[i], sorted_values[j]) > 1:
            left, right = sorted_labels[i], sorted_labels[j]
            if left > right:
                left, right = right, left
            edges.append((left, right))

    return NonCoprimalityGraphResult(
        graph=SimpleUndirectedGraph(
            vertices=sorted_labels,
            edges=tuple(edges),
        )
    )


__all__ = ["non_coprimality_graph"]
