"""Non-coprimality graph constructor."""

from __future__ import annotations

from math import gcd

from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.number_theory.non_coprimality_graph._models import (
    NonCoprimalityGraphResult,
)

__all__ = ["construct_non_coprimality_graph"]


def construct_non_coprimality_graph(
    integers: tuple[int, ...],
) -> NonCoprimalityGraphResult:
    """Construct the non-coprimality graph of a set of positive integers.

    The graph has one vertex per integer (labelled by the integer's
    canonical string) and an edge between two vertices iff their gcd > 1.
    """
    vertices = tuple(str(i) for i in integers)
    edges: list[tuple[str, str]] = []
    for i in range(len(integers)):
        for j in range(i + 1, len(integers)):
            if gcd(integers[i], integers[j]) > 1:
                a, b = str(integers[i]), str(integers[j])
                if a < b:
                    edges.append((a, b))
                else:
                    edges.append((b, a))

    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(edges),
    )
    return NonCoprimalityGraphResult(integers=integers, graph=graph)
