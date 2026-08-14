"""Graph composition and bounded nonisomorphic enumeration operations.

Two domain-atomic graph operations backed by NetworkX:

* ``graph.construct.compose`` — apply disjoint union, join, complement, or
  lexicographic product to existing simple-undirected-graph artifacts and
  materialize the result as a new graph artifact with deterministic
  a computed result.

* ``graph.enumerate.nonisomorphic`` — enumerate all nonisomorphic simple
  undirected graphs of one exact order (0-7) from the NetworkX Graph Atlas
  backend and materialize the catalog with an explicit backend boundary
  scope.  The scope artifact records that the catalog is the Graph Atlas
  representative set, not all nonisomorphic graphs of that order in
  existence.

Both operations preserve the ``jacobian.simple-undirected-graph`` payload
schema and semantics. Neither returns a mathematical conclusion or a
verification record. Construction and enumeration are deterministic NetworkX
operations; no independent checker is invoked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.contracts.graph_composition import (
    GraphCompositionOutput,
    GraphCompositionRequest,
    GraphCompositionResultArtifact,
    GraphEnumerationCandidate,
    GraphEnumerationOutput,
    GraphEnumerationRequest,
    GraphEnumerationScopeArtifact,
)
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiagnostic,
    OperationRequest,
)
from jacobian.domains._examples import example
from jacobian.graphs.artifacts import (
    GraphArtifactResources,
    graph_payload,
    load_graph_value,
    publish_graph,
)
from jacobian.graphs.atlas import graph_atlas_order, networkx_loader
from jacobian.graphs.conversions import graph_contract_from_value
from jacobian.math.graphs import (
    GraphCompositionInput,
    SimpleUndirectedGraph,
    compose_graphs,
)
from jacobian.operation_adapters import parse_operation_input
from jacobian.operation_declarations import OperationDeclaration
from jacobian.operation_errors import OperationInvocationError
from jacobian.operation_execution import execute_operation
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository

if TYPE_CHECKING:
    import networkx as nx

#: Explicit backend boundary statement for the enumeration scope.
_ENUMERATION_BACKEND_BOUNDARY = (
    "NetworkX graph_atlas_g representatives with exactly the requested order; "
    "this is the Graph Atlas catalog (orders 0-7), not all nonisomorphic "
    "graphs of that order in existence"
)


@dataclass(frozen=True)
class GraphCompositionResources:
    """Shared resources for graph composition and enumeration adapters."""

    store: ArtifactRepository
    artifacts: ArtifactService
    semantics_uri: str
    graph_schema_uri: str
    composition_result_schema_uri: str
    enumeration_scope_schema_uri: str


def register_graph_composition_resources(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    *,
    semantics_uri: str,
    graph_schema_uri: str,
) -> GraphCompositionResources:
    """Register composition schemas once for selected graph binding."""

    return GraphCompositionResources(
        store=store,
        artifacts=artifacts,
        semantics_uri=semantics_uri,
        graph_schema_uri=graph_schema_uri,
        composition_result_schema_uri=schemas.register(
            name="jacobian.graph-composition-result",
            version="1",
            schema=model_schema(GraphCompositionResultArtifact),
        ),
        enumeration_scope_schema_uri=schemas.register(
            name="jacobian.graph-enumeration-scope",
            version="1",
            schema=model_schema(GraphEnumerationScopeArtifact),
        ),
    )


def bind_selected_graph_composition(
    operation_id: str,
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    *,
    semantics_uri: str,
    graph_schema_uri: str,
    resources: GraphCompositionResources | None = None,
) -> GraphComposeAdapter | GraphEnumerateNonisomorphicAdapter | None:
    """Bind one composition or enumeration adapter without constructing both."""

    if operation_id not in {
        "graph.construct.compose",
        "graph.enumerate.nonisomorphic",
    }:
        return None
    bound = (
        resources
        if resources is not None
        else register_graph_composition_resources(
            store,
            schemas,
            artifacts,
            semantics_uri=semantics_uri,
            graph_schema_uri=graph_schema_uri,
        )
    )
    if operation_id == "graph.construct.compose":
        return GraphComposeAdapter(bound)
    return GraphEnumerateNonisomorphicAdapter(bound)


def build_graph_composition_operations(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    *,
    semantics_uri: str,
    graph_schema_uri: str,
) -> tuple[GraphComposeAdapter, GraphEnumerateNonisomorphicAdapter]:
    """Register composition and enumeration schemas and return adapters.

    This builder reuses the ``jacobian.simple-undirected-graph`` semantics
    and graph payload schema already registered in the shared graph resources.
    The caller must pass the existing ``semantics_uri`` and
    ``graph_schema_uri`` from ``GraphOperationResources``.
    """

    resources = register_graph_composition_resources(
        store,
        schemas,
        artifacts,
        semantics_uri=semantics_uri,
        graph_schema_uri=graph_schema_uri,
    )
    return (
        GraphComposeAdapter(resources),
        GraphEnumerateNonisomorphicAdapter(resources),
    )


# ---------------------------------------------------------------------------
# Graph composition
# ---------------------------------------------------------------------------


class GraphComposeAdapter:
    """Apply one graph composition operation to existing graph artifacts."""

    def __init__(self, resources: GraphCompositionResources) -> None:
        self.resources = resources
        self.spec = OperationDeclaration(
            operation_id="graph.construct.compose",
            version="1",
            request_type=GraphCompositionInput,
            result_type=SimpleUndirectedGraph,
            execute=compose_graphs,
            title="Compose graphs",
            description=(
                "Apply one deterministic graph composition operation "
                "(disjoint union, join, complement, or lexicographic product) "
                "to one or two existing simple-undirected-graph artifacts and "
                "materialize the result as a new graph artifact."
            ),
            tags=("graph", "construction", "composition"),
        )
        self._descriptor = OperationDescriptor(
            operation_id=self.spec.operation_id,
            version=self.spec.version,
            title=self.spec.title,
            description=self.spec.description,
            provider="built-in",
            input_schema=GraphCompositionRequest.model_json_schema(),
            output_schema=model_schema(GraphCompositionOutput),
            tags=self.spec.tags,
        )

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> GraphCompositionRequest:
        try:
            return parse_operation_input(GraphCompositionRequest, request.input)
        except ValidationError as exc:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="INVALID_COMPOSITION_REQUEST",
                    stage="request_validation",
                    message=("The complete graph-composition request is invalid."),
                    hint=(
                        "Provide operation, left_graph_uri, and "
                        "right_graph_uri (required for binary operations)."
                    ),
                )
            ) from exc

    def invoke(self, validated: GraphCompositionRequest) -> OperationProjection:
        graph_resources = _artifact_resources(self.resources)
        left = load_graph_value(graph_resources, validated.left_graph_uri)
        right = None
        if validated.right_graph_uri is not None:
            right = load_graph_value(graph_resources, validated.right_graph_uri)

        terminal = execute_operation(
            self.spec,
            GraphCompositionInput(
                operation=validated.operation,
                left=left,
                right=right,
            ),
        )
        if not isinstance(terminal, Completed):
            return OperationProjection(
                operation_id=self.spec.operation_id,
                version=self.spec.version,
                terminal=terminal,
            )
        backend_module = networkx_loader.get()
        backend = f"networkx.{_backend_suffix(validated.operation)}"
        result_artifact = publish_graph(
            graph_resources,
            terminal.value,
            parents=_composition_parents(validated),
            summary=f"Graph composition: {validated.operation}",
        )
        composition_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.composition_result_schema_uri,
            semantics_uri=self.resources.semantics_uri,
            payload=GraphCompositionResultArtifact(
                operation=validated.operation,
                left_graph_uri=validated.left_graph_uri,
                right_graph_uri=validated.right_graph_uri,
                result_graph_uri=result_artifact.artifact_uri,
                backend=backend,
                backend_version=backend_module.__version__,
            ).model_dump(),
            parents=(result_artifact.artifact_uri,),
            summary=f"Composition record: {validated.operation}",
        )
        output = GraphCompositionOutput(
            operation=validated.operation,
            result_graph_uri=result_artifact.artifact_uri,
            result_graph=graph_contract_from_value(terminal.value),
            composition_artifact_uri=composition_artifact.artifact_uri,
            backend=backend,
            backend_version=backend_module.__version__,
        )
        return OperationProjection(
            operation_id=self.spec.operation_id,
            version=self.spec.version,
            terminal=terminal,
            publication=PublishedOperation(
                output=output,
                artifact_uris=(
                    *_composition_parents(validated),
                    result_artifact.artifact_uri,
                    composition_artifact.artifact_uri,
                ),
            ),
        )


# ---------------------------------------------------------------------------
# Bounded nonisomorphic enumeration
# ---------------------------------------------------------------------------


class GraphEnumerateNonisomorphicAdapter:
    """Enumerate nonisomorphic graphs from the bounded NetworkX Graph Atlas."""

    def __init__(self, resources: GraphCompositionResources) -> None:
        self.resources = resources
        self._descriptor = OperationDescriptor(
            operation_id="graph.enumerate.nonisomorphic",
            version="1",
            title="Enumerate nonisomorphic graphs",
            description=(
                "Enumerate all nonisomorphic simple undirected graphs of one "
                "exact order (0-7) from the NetworkX Graph Atlas backend and "
                "materialize the catalog with an explicit backend boundary."
            ),
            provider="built-in",
            input_schema=GraphEnumerationRequest.model_json_schema(),
            output_schema=model_schema(GraphEnumerationOutput),
            tags=(
                "graph",
                "enumeration",
                "nonisomorphic",
                "bounded-search",
            ),
            examples=(
                example(
                    "order_zero",
                    "Enumerate graphs of order zero.",
                    {"order": 0, "limit": 1, "offset": 0},
                ),
            ),
        )

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> GraphEnumerationRequest:
        try:
            return parse_operation_input(GraphEnumerationRequest, request.input)
        except ValidationError as exc:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="INVALID_ENUMERATION_REQUEST",
                    stage="request_validation",
                    message=("The complete graph-enumeration request is invalid."),
                    hint="Provide order (0-7), optional limit (1-1000), and offset (>=0).",
                )
            ) from exc

    def invoke(self, validated: GraphEnumerationRequest) -> OperationProjection:
        backend_module = networkx_loader.get()
        started = time.monotonic()
        order = validated.order
        limit = validated.limit
        offset = validated.offset

        atlas_graphs = graph_atlas_order(order)
        total_count = len(atlas_graphs)
        window = atlas_graphs[offset : offset + limit]
        truncated = (offset + limit) < total_count

        scope_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.enumeration_scope_schema_uri,
            semantics_uri=self.resources.semantics_uri,
            payload=GraphEnumerationScopeArtifact(
                source="networkx.graph_atlas_g",
                backend_version=backend_module.__version__,
                order=order,
                enumerated_count=total_count,
                backend_boundary=_ENUMERATION_BACKEND_BOUNDARY,
            ).model_dump(),
            summary=f"Nonisomorphic enumeration scope: order {order}",
        )

        graphs: list[GraphEnumerationCandidate] = []
        graph_uris: list[str] = []
        for graph in window:
            payload = graph_payload(graph)
            graph_artifact = self.resources.artifacts.put(
                schema_uri=self.resources.graph_schema_uri,
                semantics_uri=self.resources.semantics_uri,
                payload=payload,
                parents=(scope_artifact.artifact_uri,),
                summary=f"Nonisomorphic graph of order {order}",
            )
            graph_uris.append(graph_artifact.artifact_uri)
            graphs.append(
                GraphEnumerationCandidate(
                    graph_uri=graph_artifact.artifact_uri,
                    graph=graph_contract_from_value(
                        SimpleUndirectedGraph.model_validate(payload)
                    ),
                    order=graph.number_of_nodes(),
                    size=graph.number_of_edges(),
                )
            )

        output = GraphEnumerationOutput(
            graphs=tuple(graphs),
            total_count=total_count,
            returned_count=len(graphs),
            truncated=truncated,
            scope_uri=scope_artifact.artifact_uri,
            backend_version=backend_module.__version__,
            backend_boundary=_ENUMERATION_BACKEND_BOUNDARY,
        )
        return OperationProjection(
            operation_id=self.descriptor.operation_id,
            version=self.descriptor.version,
            terminal=Completed(
                value=output,
                runtime_ms=_runtime_ms(started),
            ),
            publication=PublishedOperation(
                output=output,
                artifact_uris=(scope_artifact.artifact_uri, *graph_uris),
            ),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _backend_suffix(operation: str) -> str:
    if operation == "DISJOINT_UNION":
        return "disjoint_union"
    if operation == "JOIN":
        return "full_join"
    if operation == "COMPLEMENT":
        return "complement"
    if operation == "LEXICOGRAPHIC_PRODUCT":
        return "lexicographic_product"
    raise ValueError(f"unsupported composition operation: {operation}")


def _require_right_graph(
    operation: str,
    right: nx.Graph[Any] | None,
) -> nx.Graph[Any]:
    if right is None:
        raise OperationInvocationError(
            OperationDiagnostic(
                code="MISSING_RIGHT_GRAPH",
                stage="composition",
                message=f"{operation} requires a right graph.",
                hint="Provide right_graph_uri for this binary operation.",
            )
        )
    return right


def _apply_composition(
    operation: str,
    left: nx.Graph[Any],
    right: nx.Graph[Any] | None,
) -> nx.Graph[Any]:
    backend = networkx_loader.get()
    if operation == "DISJOINT_UNION":
        right = _require_right_graph(operation, right)
        return cast("nx.Graph[Any]", backend.disjoint_union(left, right))
    if operation == "JOIN":
        right = _require_right_graph(operation, right)
        return _join(left, right)
    if operation == "COMPLEMENT":
        return cast("nx.Graph[Any]", backend.complement(left))
    if operation == "LEXICOGRAPHIC_PRODUCT":
        right = _require_right_graph(operation, right)
        return cast("nx.Graph[Any]", backend.lexicographic_product(left, right))
    raise ValueError(f"unsupported composition operation: {operation}")


def _join(left: nx.Graph[Any], right: nx.Graph[Any]) -> nx.Graph[Any]:
    """Construct the graph join via the NetworkX ``full_join`` backend.

    Source graphs share the canonical ``v*`` label space, so disjoint
    ``rename`` prefixes keep the input node sets disjoint for
    ``full_join``; ``graph_payload`` re-canonicalizes the merged result.
    """
    return cast(
        "nx.Graph[Any]",
        networkx_loader.get().full_join(left, right, rename=("L", "R")),
    )


def _composition_parents(
    validated: GraphCompositionRequest,
) -> tuple[str, ...]:
    if validated.right_graph_uri is not None:
        return (validated.left_graph_uri, validated.right_graph_uri)
    return (validated.left_graph_uri,)


def _artifact_resources(
    resources: GraphCompositionResources,
) -> GraphArtifactResources:
    return GraphArtifactResources(
        store=resources.store,
        artifacts=resources.artifacts,
        semantics_uri=resources.semantics_uri,
        graph_schema_uri=resources.graph_schema_uri,
    )


def _runtime_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
