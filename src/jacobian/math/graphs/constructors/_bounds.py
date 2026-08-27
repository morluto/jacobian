"""Admission planning for exact triangle profiles."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.values import (
    SimpleUndirectedGraph,
    simple_undirected_graph_wire_bytes,
)

# The profile kernel builds adjacency once, scans common neighbors once to
# count triangles, and scans them once more to retain the exact rows. The
# graph's 256-vertex representation bound keeps this conservative work budget
# finite, while the actual edge neighborhoods determine each request's cost.
MAX_TRIANGLE_PROFILE_WORK_UNITS = 64_000_000

_TRIANGLE_ROW_FIXED_BYTES = len(
    encode_strict_json({"vertices": ["", "", ""]})
) - 3 * len(encode_strict_json(""))


@dataclass(frozen=True, slots=True)
class TriangleProfileAdmission:
    """The exact bounded scan and output plan for one triangle request."""

    triangle_indices: tuple[tuple[int, int, int], ...]
    triangle_count: int
    output_bytes: int
    work_units: int


def _triangle_indices(
    graph: SimpleUndirectedGraph,
    adjacency: tuple[frozenset[int], ...],
) -> tuple[tuple[int, int, int], ...]:
    """Return triangle indices in the graph's authoritative vertex order."""

    vertex_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    triangles: list[tuple[int, int, int]] = []
    for left, right in graph.edges:
        left_index = vertex_index[left]
        right_index = vertex_index[right]
        first, second = sorted((left_index, right_index))
        triangles.extend(
            (first, second, third)
            for third in adjacency[first] & adjacency[second]
            if third > second
        )
    return tuple(triangles)


def admit_triangle_profile(graph: SimpleUndirectedGraph) -> TriangleProfileAdmission:
    """Build one exact triangle scan and reject unrepresentable profiles."""

    vertex_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    adjacency_sets: list[set[int]] = [set() for _ in graph.vertices]
    for left, right in graph.edges:
        left_index = vertex_index[left]
        right_index = vertex_index[right]
        adjacency_sets[left_index].add(right_index)
        adjacency_sets[right_index].add(left_index)
    adjacency = tuple(frozenset(neighbors) for neighbors in adjacency_sets)

    triangle_count = 0
    triangle_label_bytes = 0
    encoded_labels = tuple(len(encode_strict_json(label)) for label in graph.vertices)
    common_neighbor_work = 0
    for left, right in graph.edges:
        left_index = vertex_index[left]
        right_index = vertex_index[right]
        first, second = sorted((left_index, right_index))
        common_neighbor_work += min(
            len(adjacency[first]),
            len(adjacency[second]),
        )
        for third in adjacency[first] & adjacency[second]:
            if third > second:
                triangle_count += 1
                triangle_label_bytes += (
                    encoded_labels[first]
                    + encoded_labels[second]
                    + encoded_labels[third]
                )

    source_bytes = simple_undirected_graph_wire_bytes(graph)
    empty_result_bytes = len(
        encode_strict_json(
            {
                "source": graph.model_dump(mode="json"),
                "triangles": [],
                "triangle_count": triangle_count,
            }
        )
    )
    output_bytes = (
        empty_result_bytes
        + _TRIANGLE_ROW_FIXED_BYTES * triangle_count
        + triangle_label_bytes
        + max(0, triangle_count - 1)
    )
    output_limit = CanonicalLimits().max_output_bytes
    if output_bytes > output_limit:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.triangle_profile.output_budget",
            message=(
                "triangle profile retains its source graph and exact labelled rows "
                f"({output_bytes:,} bytes), exceeding the {output_limit:,}-byte "
                "canonical output budget"
            ),
        )

    work_units = (
        source_bytes
        + output_bytes
        + len(graph.vertices)
        + len(graph.edges)
        + 2 * common_neighbor_work
        + 3 * triangle_count
    )
    if work_units > MAX_TRIANGLE_PROFILE_WORK_UNITS:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.triangle_profile.work_budget",
            message=(
                "triangle profile requires "
                f"{work_units:,} graph-scan and output work units, exceeding the "
                f"{MAX_TRIANGLE_PROFILE_WORK_UNITS:,}-unit bound"
            ),
        )

    return TriangleProfileAdmission(
        triangle_indices=_triangle_indices(graph, adjacency),
        triangle_count=triangle_count,
        output_bytes=output_bytes,
        work_units=work_units,
    )


__all__ = [
    "MAX_TRIANGLE_PROFILE_WORK_UNITS",
    "TriangleProfileAdmission",
    "admit_triangle_profile",
]
