"""Common-neighbour profile kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.common_neighbor_profile._models import (
    MAX_VERTICES,
    CommonNeighborProfileResult,
    CommonNeighborRow,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["compute_common_neighbor_profile"]

MAX_RESULT_BYTES = CanonicalLimits().max_output_bytes


@dataclass(frozen=True, slots=True)
class _ProfilePlan:
    rows: tuple[tuple[str, str, tuple[str, ...]], ...]


def _array_size(item_sizes: list[int]) -> int:
    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


def _reject(code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("graph",), code=f"common_neighbor.{code}", message=message
    )


def _admit_graph(
    graph: SimpleUndirectedGraph,
) -> tuple[dict[str, set[str]], _ProfilePlan]:
    if not isinstance(graph, SimpleUndirectedGraph):
        _reject("invalid_graph", "graph must be a simple undirected graph")
    vertices = list(graph.vertices)
    if len(vertices) > MAX_VERTICES:
        _reject("too_many_vertices", f"at most {MAX_VERTICES} vertices are supported")

    adjacency: dict[str, set[str]] = {vertex: set() for vertex in vertices}
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    try:
        encoded_vertex_sizes = {
            vertex: len(encode_strict_json(vertex)) for vertex in vertices
        }
        source_bytes = len(encode_strict_json(graph.model_dump(mode="json")))
        row_sizes: list[int] = []
        for index, left in enumerate(vertices):
            for right in vertices[index + 1 :]:
                common_count = 0
                common_bytes = 0
                smaller, larger = sorted((adjacency[left], adjacency[right]), key=len)
                for vertex in smaller:
                    if vertex in larger:
                        common_count += 1
                        common_bytes += encoded_vertex_sizes[vertex]
                row_sizes.append(
                    strict_json_object_size(
                        (
                            ("vertex_u", encoded_vertex_sizes[left]),
                            ("vertex_v", encoded_vertex_sizes[right]),
                            (
                                "common_neighbors",
                                2 + max(common_count - 1, 0) + common_bytes,
                            ),
                            (
                                "codegree",
                                len(encode_strict_json(common_count)),
                            ),
                        )
                    )
                )
        upper_bound = strict_json_object_size(
            (
                ("graph", source_bytes),
                ("rows", _array_size(row_sizes)),
            )
        )
    except ValueError as exc:
        _reject("source_representation", str(exc))
    if upper_bound > MAX_RESULT_BYTES:
        _reject(
            "result_size_bound",
            f"the complete profile exceeds the {MAX_RESULT_BYTES}-byte output bound",
        )

    rows = tuple(
        (
            left,
            right,
            tuple(sorted(adjacency[left] & adjacency[right])),
        )
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
    )
    return adjacency, _ProfilePlan(rows=rows)


def compute_common_neighbor_profile(
    graph: SimpleUndirectedGraph,
) -> CommonNeighborProfileResult:
    """Return the complete common-neighbour profile of a simple graph.

    For every unordered pair of distinct vertices, return the sorted
    set of common neighbours, its cardinality (codegree), in canonical
    source-vertex order.
    """
    _adjacency, plan = _admit_graph(graph)
    rows = [
        CommonNeighborRow(
            vertex_u=left,
            vertex_v=right,
            common_neighbors=common,
            codegree=len(common),
        )
        for left, right, common in plan.rows
    ]
    return CommonNeighborProfileResult(graph=graph, rows=tuple(rows))
