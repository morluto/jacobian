"""Typed wire contracts for tree-decomposition operations."""

from __future__ import annotations

import unicodedata
from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.decomposition.tree_decompositions.values import (
    TreeDecomposition,
)


def _normalized_tree_nodes(decomposition: TreeDecomposition) -> list[str]:
    return [unicodedata.normalize("NFC", node) for node in decomposition.tree_nodes]


class WidthRequest(StrictModel):
    """Compute the width of a tree decomposition."""

    decomposition: TreeDecomposition


class WidthResult(StrictModel):
    """The width and per-bag cardinalities."""

    decomposition: TreeDecomposition

    bag_sizes: tuple[int, ...]
    max_bag_cardinality: int = Field(ge=0)
    width: int = Field(ge=-1)
    maximum_bag_nodes: tuple[str, ...]

    @model_validator(mode="after")
    def bind_width(self) -> Self:
        if len(self.bag_sizes) != len(self.decomposition.tree_nodes) or not set(
            self.maximum_bag_nodes
        ) <= set(self.decomposition.tree_nodes):
            raise ValueError("width profile must use the decomposition bag axis")
        return self


class VertexOccurrencesRequest(StrictModel):
    """Compute per-source-vertex occurrence subtrees."""

    decomposition: TreeDecomposition


class OccurrenceSubtree(StrictModel):
    """Occurrence profile for one source-graph vertex."""

    vertex: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    count: int = Field(ge=0)
    leaves: tuple[str, ...]


class VertexOccurrencesResult(StrictModel):
    """Per-source-vertex occurrence subtree node set, induced tree edges,
    count, and leaf/extremal nodes."""

    decomposition: TreeDecomposition

    occurrences: tuple[OccurrenceSubtree, ...]

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        if (
            tuple(row.vertex for row in self.occurrences)
            != self.decomposition.graph.vertices
        ):
            raise ValueError("occurrences must cover the source graph vertex axis")
        nodes = set(self.decomposition.tree_nodes)
        for row in self.occurrences:
            if (
                not set(row.nodes) <= nodes
                or not set(row.leaves) <= set(row.nodes)
                or any(a not in row.nodes or b not in row.nodes for a, b in row.edges)
            ):
                raise ValueError(
                    "occurrence subtrees must use the source bag-node axis"
                )
        return self

    @property
    def per_vertex(self) -> dict[str, OccurrenceSubtree]:
        """Native mapping projection for callers that need keyed lookup."""
        return {row.vertex: row for row in self.occurrences}


class AdhesionsRequest(StrictModel):
    """Compute adhesions of a tree decomposition."""

    decomposition: TreeDecomposition


class Adhesion(StrictModel):
    edge: tuple[str, str]
    adhesion: tuple[str, ...]
    size: int = Field(ge=0)


class AdhesionsResult(StrictModel):
    """Per-tree-edge adhesion, maximum adhesion, and size profile."""

    decomposition: TreeDecomposition

    edges: tuple[Adhesion, ...]
    max_adhesion: int = Field(ge=0)
    size_profile: tuple[int, ...]

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        _require_adhesion_axes(self.decomposition, self.edges)
        if len(self.size_profile) != len(self.edges):
            raise ValueError("size profile must cover the tree edge axis")
        return self


class RerootRequest(StrictModel):
    """Reroot a tree decomposition at a selected tree node."""

    decomposition: TreeDecomposition
    root: str


class RerootNode(StrictModel):
    """One source-tree node's rooted profile."""

    node: str
    parent: str | None
    children: tuple[str, ...]
    depth: int = Field(ge=0)
    path: tuple[str, ...]


class RerootResult(StrictModel):
    """The rerooted decomposition with parent/children/depth/paths."""

    decomposition: TreeDecomposition

    root: str
    nodes: tuple[RerootNode, ...]

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        nodes = set(self.decomposition.tree_nodes)
        if (
            self.root not in nodes
            or tuple(row.node for row in self.nodes) != self.decomposition.tree_nodes
        ):
            raise ValueError("rooted profiles must cover the source bag-node axis")
        if any(
            row.parent is not None and row.parent not in nodes for row in self.nodes
        ):
            raise ValueError("rooted profile values must use source bag nodes")
        if any(
            not set(row.children) <= nodes or not set(row.path) <= nodes
            for row in self.nodes
        ):
            raise ValueError("rooted profile values must use source bag nodes")
        return self

    @property
    def parent(self) -> dict[str, str | None]:
        return {row.node: row.parent for row in self.nodes}

    @property
    def children(self) -> dict[str, tuple[str, ...]]:
        return {row.node: row.children for row in self.nodes}

    @property
    def depth(self) -> dict[str, int]:
        return {row.node: row.depth for row in self.nodes}

    @property
    def paths(self) -> dict[str, tuple[str, ...]]:
        return {row.node: row.path for row in self.nodes}


class RestrictRequest(StrictModel):
    """Restrict a tree decomposition to a source-vertex subset."""

    decomposition: TreeDecomposition
    subset: tuple[str, ...] = Field(min_length=1)


class BagIntersectionGraphRequest(StrictModel):
    """Compute the weighted bag-intersection graph of a decomposition."""

    decomposition: TreeDecomposition


class BagNode(StrictModel):
    node: str
    bag_size: int = Field(ge=0)


class BagIntersectionGraphResult(StrictModel):
    """The weighted tree: each node labelled by bag size, each edge by adhesion."""

    decomposition: TreeDecomposition

    nodes: tuple[BagNode, ...]
    edges: tuple[Adhesion, ...]
    max_adhesion: int = Field(ge=0)

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        if tuple(node.node for node in self.nodes) != self.decomposition.tree_nodes:
            raise ValueError("bag graph must retain the source bag-node axis")
        _require_adhesion_axes(self.decomposition, self.edges)
        return self


def _require_adhesion_axes(
    source: TreeDecomposition, edges: tuple[Adhesion, ...]
) -> None:
    expected = tuple(tuple(sorted(edge)) for edge in source.tree_edges)
    if tuple(row.edge for row in edges) != expected:
        raise ValueError("adhesions must retain the source tree-edge axis")
    if any(not set(row.adhesion) <= set(source.graph.vertices) for row in edges):
        raise ValueError("adhesions must use source graph vertices")


__all__ = [
    "Adhesion",
    "AdhesionsRequest",
    "AdhesionsResult",
    "BagIntersectionGraphRequest",
    "BagIntersectionGraphResult",
    "BagNode",
    "OccurrenceSubtree",
    "RerootNode",
    "RerootRequest",
    "RerootResult",
    "RestrictRequest",
    "VertexOccurrencesRequest",
    "VertexOccurrencesResult",
    "WidthRequest",
    "WidthResult",
]
