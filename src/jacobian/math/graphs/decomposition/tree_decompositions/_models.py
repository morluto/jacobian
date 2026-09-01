"""Typed wire contracts for tree-decomposition operations."""

from __future__ import annotations

import unicodedata
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

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

    bag_sizes: tuple[int, ...]
    max_bag_cardinality: int = Field(ge=1)
    width: int = Field(ge=0)
    maximum_bag_nodes: tuple[str, ...]

    @model_validator(mode="after")
    def bind_width(self) -> Self:
        if self.max_bag_cardinality != self.width + 1:
            raise PydanticCustomError(
                "graph.width_must_equal_max_bag_cardinality_minus_one",
                "width must equal max_bag_cardinality minus one",
            )
        return self


class VertexOccurrencesRequest(StrictModel):
    """Compute per-source-vertex occurrence subtrees."""

    decomposition: TreeDecomposition


class OccurrenceSubtree(StrictModel):
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    count: int = Field(ge=0)
    leaves: tuple[str, ...]


class VertexOccurrencesResult(StrictModel):
    """Per-source-vertex occurrence subtree node set, induced tree edges,
    count, and leaf/extremal nodes."""

    per_vertex: dict[str, OccurrenceSubtree]


class AdhesionsRequest(StrictModel):
    """Compute adhesions of a tree decomposition."""

    decomposition: TreeDecomposition


class Adhesion(StrictModel):
    edge: tuple[str, str]
    adhesion: tuple[str, ...]
    size: int = Field(ge=0)


class AdhesionsResult(StrictModel):
    """Per-tree-edge adhesion, maximum adhesion, and size profile."""

    edges: tuple[Adhesion, ...]
    max_adhesion: int = Field(ge=0)
    size_profile: tuple[int, ...]


class RerootRequest(StrictModel):
    """Reroot a tree decomposition at a selected tree node."""

    decomposition: TreeDecomposition
    root: str


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


class BagIntersectionGraphRequest(StrictModel):
    """Compute the weighted bag-intersection graph of a decomposition."""

    decomposition: TreeDecomposition


class BagNode(StrictModel):
    node: str
    bag_size: int = Field(ge=0)


class BagIntersectionGraphResult(StrictModel):
    """The weighted tree: each node labelled by bag size, each edge by adhesion."""

    nodes: tuple[BagNode, ...]
    edges: tuple[Adhesion, ...]
    max_adhesion: int = Field(ge=0)


__all__ = [
    "Adhesion",
    "AdhesionsRequest",
    "AdhesionsResult",
    "BagIntersectionGraphRequest",
    "BagIntersectionGraphResult",
    "BagNode",
    "OccurrenceSubtree",
    "RerootRequest",
    "RerootResult",
    "RestrictRequest",
    "VertexOccurrencesRequest",
    "VertexOccurrencesResult",
    "WidthRequest",
    "WidthResult",
]
