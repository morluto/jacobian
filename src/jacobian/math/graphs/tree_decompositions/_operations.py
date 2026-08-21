"""Domain adapter for tree-decomposition operations."""

from __future__ import annotations

from jacobian.math.graphs.tree_decompositions._models import (
    AdhesionsRequest,
    AdhesionsResult,
    BagIntersectionGraphRequest,
    BagIntersectionGraphResult,
    RerootRequest,
    RerootResult,
    RestrictRequest,
    RestrictResult,
    VertexOccurrencesRequest,
    VertexOccurrencesResult,
    WidthRequest,
    WidthResult,
)
from jacobian.math.graphs.tree_decompositions.operations import (
    adhesions,
    bag_intersection_graph,
    reroot,
    restrict,
    vertex_occurrences,
    width,
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
    result = width(request.decomposition)
    return WidthResult(
        bag_sizes=result["bag_sizes"],  # type: ignore[arg-type]
        max_bag_cardinality=result["max_bag_cardinality"],  # type: ignore[arg-type]
        width=result["width"],  # type: ignore[arg-type]
        maximum_bag_nodes=result["maximum_bag_nodes"],  # type: ignore[arg-type]
    )


def compute_vertex_occurrences(
    request: VertexOccurrencesRequest,
) -> VertexOccurrencesResult:
    result = vertex_occurrences(request.decomposition)
    return VertexOccurrencesResult(per_vertex=result)


def compute_adhesions(request: AdhesionsRequest) -> AdhesionsResult:
    result = adhesions(request.decomposition)
    return AdhesionsResult(
        edges=result["edges"],  # type: ignore[arg-type]
        max_adhesion=result["max_adhesion"],  # type: ignore[arg-type]
        size_profile=result["size_profile"],  # type: ignore[arg-type]
    )


def compute_reroot(request: RerootRequest) -> RerootResult:
    result = reroot(request.decomposition, request.root)
    return RerootResult(
        root=result["root"],  # type: ignore[arg-type]
        parent=result["parent"],  # type: ignore[arg-type]
        children=result["children"],  # type: ignore[arg-type]
        depth=result["depth"],  # type: ignore[arg-type]
        paths=result["paths"],  # type: ignore[arg-type]
    )


def compute_restrict(request: RestrictRequest) -> RestrictResult:
    result = restrict(request.decomposition, frozenset(request.subset))
    return RestrictResult(
        graph=result["graph"],  # type: ignore[arg-type]
        tree_nodes=result["tree_nodes"],  # type: ignore[arg-type]
        tree_edges=result["tree_edges"],  # type: ignore[arg-type]
        bags=result["bags"],  # type: ignore[arg-type]
    )


def compute_bag_intersection_graph(
    request: BagIntersectionGraphRequest,
) -> BagIntersectionGraphResult:
    result = bag_intersection_graph(request.decomposition)
    return BagIntersectionGraphResult(
        nodes=result["nodes"],  # type: ignore[arg-type]
        edges=result["edges"],  # type: ignore[arg-type]
        max_adhesion=result["max_adhesion"],  # type: ignore[arg-type]
    )
