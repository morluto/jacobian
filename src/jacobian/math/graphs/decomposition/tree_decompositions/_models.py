"""Typed wire contracts for tree-decomposition operations."""

from __future__ import annotations

import unicodedata
from collections import deque
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.math.graphs.decomposition.tree_decompositions.values import (
    TreeDecomposition,
)


def _json_array_size(item_sizes: list[int]) -> int:
    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


def _normalized_tree_nodes(decomposition: TreeDecomposition) -> list[str]:
    return [unicodedata.normalize("NFC", node) for node in decomposition.tree_nodes]


def _reroot_result_wire_bytes(decomposition: TreeDecomposition, root: str) -> int:
    """Return the exact strict-JSON size of ``reroot(decomposition, root)``.

    The result's only superlinear field is the map of root-to-node paths.  Its
    exact encoded size follows from one traversal and one accumulated array
    size per node, without constructing or retaining every repeated path
    label before admission.  Labels are measured after NFC normalization,
    matching how the canonical transport boundary normalizes string values.
    """

    node_index = {node: index for index, node in enumerate(decomposition.tree_nodes)}
    root_index = node_index[root]
    adjacency: list[list[int]] = [[] for _ in decomposition.tree_nodes]
    for left, right in decomposition.tree_edges:
        left_index = node_index[left]
        right_index = node_index[right]
        adjacency[left_index].append(right_index)
        adjacency[right_index].append(left_index)

    parent: list[int | None] = [None] * len(decomposition.tree_nodes)
    depth = [0] * len(decomposition.tree_nodes)
    traversal = [root_index]
    queue = deque([root_index])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor == parent[current] or neighbor == root_index:
                continue
            parent[neighbor] = current
            depth[neighbor] = depth[current] + 1
            traversal.append(neighbor)
            queue.append(neighbor)

    normalized_nodes = _normalized_tree_nodes(decomposition)
    encoded_nodes = [len(encode_strict_json(node)) for node in normalized_nodes]
    parent_fields = []
    children: list[list[int]] = [[] for _ in decomposition.tree_nodes]
    for index in traversal:
        parent_index = parent[index]
        parent_fields.append(
            (
                normalized_nodes[index],
                4 if parent_index is None else encoded_nodes[parent_index],
            )
        )
        if parent_index is not None:
            children[parent_index].append(index)
    children_size = strict_json_object_size(
        (
            normalized_nodes[index],
            _json_array_size([encoded_nodes[child] for child in children[index]]),
        )
        for index in range(len(decomposition.tree_nodes))
    )
    depth_size = strict_json_object_size(
        (normalized_nodes[index], len(str(depth[index]))) for index in traversal
    )
    path_sizes = [0] * len(decomposition.tree_nodes)
    path_sizes[root_index] = _json_array_size([encoded_nodes[root_index]])
    for index in traversal[1:]:
        parent_index = parent[index]
        assert parent_index is not None
        path_sizes[index] = path_sizes[parent_index] + 1 + encoded_nodes[index]
    paths_size = strict_json_object_size(
        (normalized_nodes[index], path_sizes[index]) for index in traversal
    )
    return strict_json_object_size(
        (
            ("root", encoded_nodes[root_index]),
            ("parent", strict_json_object_size(parent_fields)),
            ("children", children_size),
            ("depth", depth_size),
            ("paths", paths_size),
        )
    )


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

    @model_validator(mode="after")
    def require_valid_root(self) -> Self:
        if self.root not in self.decomposition.tree_nodes:
            raise PydanticCustomError(
                "graph.root_must_be_a_declared_tree_node",
                "root must be a declared tree node",
            )
        return self

    @model_validator(mode="after")
    def require_transportable_result(self) -> Self:
        """Preflight the owner-local root-to-node path projection.

        A tree with at most 256 nodes can still produce quadratically many
        repeated node labels across its root-to-node paths.  Compute that
        deterministic projection while admitting the request, so the final
        transport wrapper never discovers an oversized result after execution.
        Labels are compared after the transport boundary's NFC normalization,
        so canonically equivalent spellings cannot collide as result keys.
        """

        normalized_nodes = _normalized_tree_nodes(self.decomposition)
        if len(set(normalized_nodes)) != len(normalized_nodes):
            raise PydanticCustomError(
                "graph.reroot_tree_node_labels_collide_after_normalization",
                "tree node labels collide after Unicode NFC normalization",
            )
        result_bytes = _reroot_result_wire_bytes(self.decomposition, self.root)
        output_limit = CanonicalLimits().max_output_bytes
        if result_bytes > output_limit:
            raise PydanticCustomError(
                "graph.reroot_result_exceeds_transport_limit",
                "rerooted tree-decomposition paths exceed the "
                f"{output_limit}-byte canonical output limit",
            )
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

    @model_validator(mode="after")
    def require_source_vertices(self) -> Self:
        source_vertices = set(self.decomposition.graph.vertices)
        if not set(self.subset).issubset(source_vertices):
            raise PydanticCustomError(
                "graph.subset_must_contain_only_declared_source_vertice",
                "subset must contain only declared source vertices",
            )
        return self


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
