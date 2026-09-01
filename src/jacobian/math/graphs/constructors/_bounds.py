"""Admission planning for exact triangle profiles."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.values import SimpleUndirectedGraph

# The profile kernel builds adjacency once and scans common neighbors once to
# count and retain the exact rows. The
# graph's 256-vertex representation bound keeps this conservative work budget
# finite, while the actual edge neighborhoods determine each request's cost.
MAX_TRIANGLE_PROFILE_WORK_UNITS = 64_000_000
MAX_TRIANGLE_PROFILE_ROWS = 1_000_000
MAX_TRIANGLE_PROFILE_RETAINED_LABEL_CHARACTERS = 100_000_000


@dataclass(frozen=True, slots=True)
class TriangleProfileAdmission:
    """The exact bounded scan and output plan for one triangle request."""

    triangle_indices: tuple[tuple[int, int, int], ...]
    triangle_count: int
    work_units: int


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

    triangle_indices: list[tuple[int, int, int]] = []
    common_neighbor_work = 0
    retained_label_characters = sum(len(vertex) for vertex in graph.vertices) + sum(
        len(left) + len(right) for left, right in graph.edges
    )
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
                triangle_indices.append((first, second, third))
                retained_label_characters += (
                    len(graph.vertices[first])
                    + len(graph.vertices[second])
                    + len(graph.vertices[third])
                )
                if (
                    retained_label_characters
                    > MAX_TRIANGLE_PROFILE_RETAINED_LABEL_CHARACTERS
                ):
                    raise OperationDomainValidationError(
                        location=("graph",),
                        code="graph.triangle_profile.retained_labels_exceed_bound",
                        message=(
                            "triangle profile exceeds the retained "
                            "label-character bound"
                        ),
                    )
    triangle_count = len(triangle_indices)
    work_units = (
        len(graph.vertices)
        + len(graph.edges)
        + common_neighbor_work
        + 3 * triangle_count
    )
    if triangle_count > MAX_TRIANGLE_PROFILE_ROWS:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.triangle_profile.row_bound",
            message=(
                f"triangle profile has {triangle_count:,} rows, exceeding the "
                f"{MAX_TRIANGLE_PROFILE_ROWS:,}-row materialization bound"
            ),
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
        triangle_indices=tuple(triangle_indices),
        triangle_count=triangle_count,
        work_units=work_units,
    )


__all__ = [
    "MAX_TRIANGLE_PROFILE_RETAINED_LABEL_CHARACTERS",
    "MAX_TRIANGLE_PROFILE_ROWS",
    "MAX_TRIANGLE_PROFILE_WORK_UNITS",
    "TriangleProfileAdmission",
    "admit_triangle_profile",
]
