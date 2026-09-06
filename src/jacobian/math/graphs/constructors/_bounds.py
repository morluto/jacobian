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
    """Reusable derived facts for one admitted triangle request.

    Carries adjacency and degree data for the kernel's single enumeration.
    Triangle rows are computed once by the operation, not during admission.
    """

    vertex_index: dict[str, int]
    adjacency: tuple[frozenset[int], ...]
    work_estimate: int


def _admit_dimension(dimension: int, *, maximum: int, operation: str) -> None:
    """Reject a construction dimension outside its admitted envelope."""
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise OperationDomainValidationError(
            location=("dimension",),
            code=f"graph.constructors.{operation}.dimension_type",
            message="dimension must be an integer",
        )
    if not 0 <= dimension <= maximum:
        raise OperationDomainValidationError(
            location=("dimension",),
            code=f"graph.constructors.{operation}.dimension_bound",
            message=f"dimension must be between 0 and {maximum}",
        )


def admit_hypercube_dimension(dimension: int) -> None:
    """Admit a hypercube dimension before construction."""
    from jacobian.math.graphs.constructors._models import MAX_HYPERCUBE_DIMENSION

    _admit_dimension(dimension, maximum=MAX_HYPERCUBE_DIMENSION, operation="hypercube")


def admit_keller_dimension(dimension: int) -> None:
    """Admit a Keller dimension before construction."""
    from jacobian.math.graphs.constructors._models import MAX_KELLER_DIMENSION

    _admit_dimension(dimension, maximum=MAX_KELLER_DIMENSION, operation="keller")


def admit_triangle_profile(graph: SimpleUndirectedGraph) -> TriangleProfileAdmission:
    """Admit request representation and bound work without enumerating rows."""

    vertex_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    adjacency_sets: list[set[int]] = [set() for _ in graph.vertices]
    for left, right in graph.edges:
        left_index = vertex_index[left]
        right_index = vertex_index[right]
        adjacency_sets[left_index].add(right_index)
        adjacency_sets[right_index].add(left_index)
    adjacency = tuple(frozenset(neighbors) for neighbors in adjacency_sets)

    # Estimate scan work from degrees only; the kernel enumerates rows once.
    # For edge (u, v), common-neighbor work is bounded by min(deg u, deg v),
    # and each triangle contributes 3 to that sum, so the estimate upper-bounds
    # both scan work and 3 * row count without materializing any triangle.
    common_neighbor_work = 0
    for left, right in graph.edges:
        left_index = vertex_index[left]
        right_index = vertex_index[right]
        first, second = sorted((left_index, right_index))
        common_neighbor_work += min(
            len(adjacency[first]),
            len(adjacency[second]),
        )
    work_estimate = len(graph.vertices) + len(graph.edges) + common_neighbor_work
    # Every retained triangle requires three edge-neighbor checks.  The degree
    # estimate bounds those checks before the kernel constructs any result row.
    triangle_row_upper_bound = common_neighbor_work // 3
    if triangle_row_upper_bound > MAX_TRIANGLE_PROFILE_ROWS:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.triangle_profile.row_bound",
            message=(
                "triangle profile can require up to "
                f"{triangle_row_upper_bound:,} rows, exceeding the "
                f"{MAX_TRIANGLE_PROFILE_ROWS:,}-row materialization bound"
            ),
        )
    if work_estimate > MAX_TRIANGLE_PROFILE_WORK_UNITS:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.triangle_profile.work_budget",
            message=(
                "triangle profile requires "
                f"{work_estimate:,} graph-scan work units, exceeding the "
                f"{MAX_TRIANGLE_PROFILE_WORK_UNITS:,}-unit bound"
            ),
        )
    base_labels = sum(len(vertex) for vertex in graph.vertices) + sum(
        len(left) + len(right) for left, right in graph.edges
    )
    if base_labels > MAX_TRIANGLE_PROFILE_RETAINED_LABEL_CHARACTERS:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.triangle_profile.retained_labels_exceed_bound",
            message=("triangle profile exceeds the retained label-character bound"),
        )

    return TriangleProfileAdmission(
        vertex_index=vertex_index,
        adjacency=adjacency,
        work_estimate=work_estimate,
    )


__all__ = [
    "MAX_TRIANGLE_PROFILE_RETAINED_LABEL_CHARACTERS",
    "MAX_TRIANGLE_PROFILE_ROWS",
    "MAX_TRIANGLE_PROFILE_WORK_UNITS",
    "TriangleProfileAdmission",
    "admit_hypercube_dimension",
    "admit_keller_dimension",
    "admit_triangle_profile",
]
