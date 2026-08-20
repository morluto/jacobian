"""Typed wire contracts for tree-decomposition operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.tree_decompositions.values import TreeDecomposition


class WidthRequest(StrictModel):
    """Compute the width of a tree decomposition."""

    decomposition: TreeDecomposition


class WidthResult(StrictModel):
    """The width and per-bag cardinalities."""

    bag_sizes: tuple[int, ...]
    max_bag_cardinality: int = Field(ge=1)
    width: int = Field(ge=0)
    maximum_bag_nodes: tuple[str, ...]

    @model_validator(mode="after")
    def bind_width(self) -> Self:
        if self.max_bag_cardinality != self.width + 1:
            raise ValueError("width must equal max_bag_cardinality minus one")
        return self


class VertexOccurrencesRequest(StrictModel):
    """Compute per-source-vertex occurrence subtrees."""

    decomposition: TreeDecomposition


class VertexOccurrencesResult(StrictModel):
    """Per-source-vertex occurrence subtree node set, induced tree edges,
    count, and leaf/extremal nodes."""

    per_vertex: dict[str, dict[str, object]]


class AdhesionsRequest(StrictModel):
    """Compute adhesions of a tree decomposition."""

    decomposition: TreeDecomposition


class AdhesionsResult(StrictModel):
    """Per-tree-edge adhesion, maximum adhesion, and size profile."""

    edges: tuple[dict[str, object], ...]
    max_adhesion: int = Field(ge=0)
    size_profile: tuple[int, ...]


class RerootRequest(StrictModel):
    """Reroot a tree decomposition at a selected tree node."""

    decomposition: TreeDecomposition
    root: str

    @model_validator(mode="after")
    def require_valid_root(self) -> Self:
        if self.root not in self.decomposition.tree_nodes:
            raise ValueError("root must be a declared tree node")
        return self


class RerootResult(StrictModel):
    """The rerooted decomposition with parent/children/depth/paths."""

    root: str
    parent: dict[str, str | None]
    children: dict[str, tuple[str, ...]]
    depth: dict[str, int]
    paths: dict[str, list[str]]


class RestrictRequest(StrictModel):
    """Restrict a tree decomposition to a source-vertex subset."""

    decomposition: TreeDecomposition
    subset: tuple[str, ...] = Field(min_length=1)


class RestrictResult(StrictModel):
    """The restricted decomposition bound to the induced source graph."""

    graph: dict[str, object]
    tree_nodes: tuple[str, ...]
    tree_edges: tuple[tuple[str, str], ...]
    bags: tuple[tuple[str, ...], ...]


class BagIntersectionGraphRequest(StrictModel):
    """Compute the weighted bag-intersection graph of a decomposition."""

    decomposition: TreeDecomposition


class BagIntersectionGraphResult(StrictModel):
    """The weighted tree: each node labelled by bag size, each edge by adhesion."""

    nodes: tuple[dict[str, object], ...]
    edges: tuple[dict[str, object], ...]
    max_adhesion: int = Field(ge=0)


__all__ = [
    "AdhesionsRequest",
    "AdhesionsResult",
    "BagIntersectionGraphRequest",
    "BagIntersectionGraphResult",
    "RerootRequest",
    "RerootResult",
    "RestrictRequest",
    "RestrictResult",
    "VertexOccurrencesRequest",
    "VertexOccurrencesResult",
    "WidthRequest",
    "WidthResult",
]
