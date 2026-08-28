"""Domain adapter for tree-decomposition operations."""

from __future__ import annotations

from jacobian.math.graphs.decomposition.tree_decompositions._models import (
    AdhesionsRequest,
    AdhesionsResult,
    BagIntersectionGraphRequest,
    BagIntersectionGraphResult,
    RerootRequest,
    RerootResult,
    RestrictRequest,
    VertexOccurrencesRequest,
    VertexOccurrencesResult,
    WidthRequest,
    WidthResult,
)
from jacobian.math.graphs.decomposition.tree_decompositions.operations import (
    adhesions,
    bag_intersection_graph,
    reroot,
    restrict,
    vertex_occurrences,
    width,
)
from jacobian.math.graphs.decomposition.tree_decompositions.values import (
    TreeDecomposition,
)

__all__ = [
    "compute_adhesions",
    "compute_bag_intersection_graph",
    "compute_reroot",
    "compute_restrict",
    "compute_vertex_occurrences",
    "compute_width",
]


def compute_width(request: WidthRequest) -> WidthResult:
    return width(request.decomposition)


def compute_vertex_occurrences(
    request: VertexOccurrencesRequest,
) -> VertexOccurrencesResult:
    return vertex_occurrences(request.decomposition)


def compute_adhesions(request: AdhesionsRequest) -> AdhesionsResult:
    return adhesions(request.decomposition)


def compute_reroot(request: RerootRequest) -> RerootResult:
    return reroot(request.decomposition, request.root)


def compute_restrict(request: RestrictRequest) -> TreeDecomposition:
    return restrict(request.decomposition, frozenset(request.subset))


def compute_bag_intersection_graph(
    request: BagIntersectionGraphRequest,
) -> BagIntersectionGraphResult:
    return bag_intersection_graph(request.decomposition)
