"""Exact non-coprimality conflict-graph construction."""

from __future__ import annotations

from itertools import combinations
from math import gcd as exact_gcd

from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    parse_canonical_integer,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.number_theory.non_coprimality_graph._models import (
    MAX_NON_COPRIMALITY_GRAPH_VERTICES,
    NonCoprimalityGraphResult,
)

MAX_GCD_DIGIT_WORK = 10_000_000


def _array_size(item_sizes: list[int]) -> int:
    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


def _base_result_bytes(label_sizes: list[int]) -> int:
    graph_size = strict_json_object_size(
        (
            ("vertices", _array_size(label_sizes)),
            ("edges", _array_size([])),
        )
    )
    return strict_json_object_size((("graph", graph_size),))


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
        value = parse_canonical_integer(label)
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

    gcd_digit_work = sum(
        min(len(sorted_labels[i]), len(sorted_labels[j]))
        for i, j in combinations(range(n), 2)
    )
    if gcd_digit_work > MAX_GCD_DIGIT_WORK:
        raise OperationDomainValidationError(
            location=("elements",),
            code="number_theory.non_coprimality_graph.work_bound_exceeded",
            message="the pairwise GCD work exceeds the operation bound",
        )
    label_sizes = [len(encode_strict_json(label)) for label in sorted_labels]
    result_bytes = _base_result_bytes(label_sizes)
    if result_bytes > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("elements",),
            code="number_theory.non_coprimality_graph.result_size_bound",
            message="the complete graph exceeds the canonical output bound",
        )

    edges: list[tuple[str, str]] = []
    for i, j in combinations(range(n), 2):
        if exact_gcd(sorted_values[i], sorted_values[j]) > 1:
            result_bytes += _array_size([label_sizes[i], label_sizes[j]])
            if edges:
                result_bytes += 1
            if result_bytes > CanonicalLimits().max_output_bytes:
                raise OperationDomainValidationError(
                    location=("elements",),
                    code="number_theory.non_coprimality_graph.result_size_bound",
                    message="the complete graph exceeds the canonical output bound",
                )
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
