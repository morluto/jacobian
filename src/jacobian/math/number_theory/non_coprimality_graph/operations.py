"""Non-coprimality graph constructor."""

from __future__ import annotations

from math import gcd

from pydantic_core import PydanticCustomError

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.number_theory.non_coprimality_graph._models import (
    NonCoprimalityGraphResult,
    _validate_integer_source,
)

__all__ = ["construct_non_coprimality_graph"]


def construct_non_coprimality_graph(
    integers: tuple[int, ...],
) -> NonCoprimalityGraphResult:
    """Construct the non-coprimality graph of a set of positive integers.

    The graph has one vertex per integer (labelled by the integer's
    canonical string) and an edge between two vertices iff their gcd > 1.
    """
    if not isinstance(integers, tuple) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in integers
    ):
        raise OperationDomainValidationError(
            location=("integers",),
            code="non_coprimality.integer_type",
            message="integers must be a tuple of integers",
        )
    vertices = tuple(format_canonical_integer(i) for i in integers)
    try:
        _validate_integer_source(vertices)
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=("integers",), code=error.type, message=str(error)
        ) from error
    sorted_pairs = sorted(
        zip(vertices, integers, strict=True), key=lambda pair: pair[1]
    )
    sorted_vertices = tuple(label for label, _ in sorted_pairs)
    edges: list[tuple[str, str]] = []
    for i in range(len(sorted_pairs)):
        for j in range(i + 1, len(sorted_pairs)):
            if gcd(sorted_pairs[i][1], sorted_pairs[j][1]) > 1:
                a, b = sorted_pairs[i][0], sorted_pairs[j][0]
                if a < b:
                    edges.append((a, b))
                else:
                    edges.append((b, a))

    graph = SimpleUndirectedGraph(
        vertices=sorted_vertices,
        edges=tuple(edges),
    )
    return NonCoprimalityGraphResult(integers=vertices, graph=graph)
