"""Typed contracts for graph composition and bounded nonisomorphic enumeration.

These contracts cover two domain-atomic graph operations:

* ``graph.construct.compose`` — apply one binary or unary graph composition
  operation (disjoint union, join, complement, lexicographic product) to
  existing simple-undirected-graph artifacts and materialize the result as a
  new graph artifact.

* ``graph.enumerate.nonisomorphic`` — enumerate all nonisomorphic simple
  undirected graphs of one exact order from a bounded backend (NetworkX Graph
  Atlas, orders 0-7) and materialize the catalog with an explicit backend
  boundary scope.

Both contracts preserve the existing ``jacobian.simple-undirected-graph``
semantics and payload schema.  Neither contract carries a mathematical
conclusion or verification record; construction and enumeration produce
computed results only.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.common import ArtifactUri, Sha256Digest
from jacobian.contracts.graph_coloring import GraphVertex
from jacobian.contracts.graph_isomorphism import SimpleUndirectedGraph
from jacobian.contracts.results import ContractModel

#: Operations supported by ``graph.construct.compose``.
CompositionOperation = Literal[
    "DISJOINT_UNION",
    "JOIN",
    "COMPLEMENT",
    "LEXICOGRAPHIC_PRODUCT",
]

#: Unary operations that consume only the left graph.
_UNARY_OPERATIONS: frozenset[str] = frozenset({"COMPLEMENT"})


class GraphExplicitConstructionRequest(ContractModel):
    """One bounded explicit simple-undirected-graph materialization request.

    Edge orientation and input order are immaterial. The producer emits one
    canonical payload with sorted vertices and sorted ascending endpoint pairs.
    """

    vertices: tuple[GraphVertex, ...] = Field(max_length=256)
    edges: tuple[tuple[GraphVertex, GraphVertex], ...] = Field(max_length=32640)

    @model_validator(mode="after")
    def require_bounded_simple_graph(self) -> Self:
        vertex_set = set(self.vertices)
        if len(vertex_set) != len(self.vertices):
            raise ValueError("graph vertices must be unique")
        if any(left == right for left, right in self.edges):
            raise ValueError("graph edges must not contain self-loops")
        if any(
            left not in vertex_set or right not in vertex_set
            for left, right in self.edges
        ):
            raise ValueError("graph edges must reference declared vertices")
        normalized = {tuple(sorted(edge)) for edge in self.edges}
        if len(normalized) != len(self.edges):
            raise ValueError("graph edges must be unique ignoring orientation")
        return self


class GraphExplicitConstructionOutput(ContractModel):
    """Canonical graph artifact produced from one completely validated request."""

    graph_uri: ArtifactUri
    graph_object_digest: Sha256Digest
    graph_schema_uri: ArtifactUri
    graph_semantics_uri: ArtifactUri
    graph: SimpleUndirectedGraph
    order: StrictInt = Field(ge=0, le=256)
    size: StrictInt = Field(ge=0, le=32640)


class GraphCompositionRequest(ContractModel):
    """Request model for ``graph.construct.compose``.

    ``left_graph_uri`` is always required.  ``right_graph_uri`` is required
    for binary operations (``DISJOINT_UNION``, ``JOIN``,
    ``LEXICOGRAPHIC_PRODUCT``) and must be absent for the unary
    ``COMPLEMENT``.
    """

    operation: CompositionOperation
    left_graph_uri: ArtifactUri
    right_graph_uri: ArtifactUri | None = None

    @model_validator(mode="after")
    def require_right_graph_for_binary_operations(self) -> Self:
        if self.operation in _UNARY_OPERATIONS:
            if self.right_graph_uri is not None:
                raise ValueError("right_graph_uri must be absent for unary operations")
        elif self.right_graph_uri is None:
            raise ValueError(f"right_graph_uri is required for {self.operation}")
        return self


class GraphCompositionResultArtifact(ContractModel):
    """Materialized artifact recording one deterministic graph composition."""

    composition_schema_version: Literal["1"] = "1"
    operation: CompositionOperation
    left_graph_uri: ArtifactUri
    right_graph_uri: ArtifactUri | None = None
    result_graph_uri: ArtifactUri
    backend: str = Field(min_length=1, max_length=128)
    backend_version: str = Field(min_length=1, max_length=64)


class GraphCompositionOutput(ContractModel):
    """Public v1 projection of one materialized graph composition."""

    operation: CompositionOperation
    result_graph_uri: ArtifactUri
    result_graph: SimpleUndirectedGraph
    composition_artifact_uri: ArtifactUri
    backend: str = Field(min_length=1, max_length=128)
    backend_version: str = Field(min_length=1, max_length=64)


class GraphEnumerationRequest(ContractModel):
    """Request model for ``graph.enumerate.nonisomorphic``.

    ``order`` is bounded by the NetworkX Graph Atlas backend (0-7).
    ``limit`` and ``offset`` control the returned window of the full catalog.
    """

    order: int = Field(ge=0, le=7)
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class GraphEnumerationScopeArtifact(ContractModel):
    """Scope artifact recording the bounded enumeration backend boundary.

    ``backend_boundary`` states explicitly what the enumeration covers so
    that agents do not mistake the Graph Atlas catalog for all nonisomorphic
    graphs of the requested order in existence.
    """

    enumeration_schema_version: Literal["1"] = "1"
    source: str = Field(min_length=1, max_length=128)
    backend_version: str = Field(min_length=1, max_length=64)
    order: int = Field(ge=0, le=7)
    enumerated_count: int = Field(ge=0)
    backend_boundary: str = Field(min_length=1, max_length=512)
