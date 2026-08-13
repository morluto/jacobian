"""Canonical graph artifacts and the shared graph storage boundary."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.contracts.artifacts import ArtifactPutResult
from jacobian.contracts.graph_isomorphism import (
    SimpleUndirectedGraph as SimpleUndirectedGraphContract,
)
from jacobian.contracts.operations import OperationDiagnostic
from jacobian.graphs.atlas import networkx_loader
from jacobian.graphs.conversions import (
    graph_contract_from_value,
    graph_value_from_contract,
)
from jacobian.math.graphs import SimpleUndirectedGraph
from jacobian.operation_errors import OperationInvocationError
from jacobian.schema_registry import model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository

if TYPE_CHECKING:
    import networkx as nx_type


ARTIFACT_URI_PATTERN = r"^artifact://sha256/[0-9a-f]{64}$"

GRAPH_PAYLOAD_SCHEMA: dict[str, Any] = model_schema(SimpleUndirectedGraphContract)


@dataclass(frozen=True, slots=True)
class GraphArtifactResources:
    """The common storage and semantics needed by graph outcomes."""

    store: ArtifactRepository
    artifacts: ArtifactService
    semantics_uri: str
    graph_schema_uri: str


def nx() -> Any:
    """Load NetworkX only when a graph operation is invoked."""

    return networkx_loader.get()


def graph_payload(graph: nx_type.Graph[Any]) -> dict[str, Any]:
    """Serialize a backend graph into the canonical graph artifact payload."""

    labels = {node: f"v{index}" for index, node in enumerate(sorted(graph.nodes))}
    edges = sorted(
        [labels[source], labels[target]]
        if labels[source] < labels[target]
        else [labels[target], labels[source]]
        for source, target in graph.edges
    )
    return {
        "graph_schema_version": "1",
        "vertices": [labels[node] for node in sorted(graph.nodes)],
        "edges": edges,
    }


def publish_graph(
    resources: GraphArtifactResources,
    graph: SimpleUndirectedGraph,
    *,
    parents: tuple[str, ...] = (),
    summary: str,
) -> ArtifactPutResult:
    """Publish one graph value without changing its mathematical identity."""

    return resources.artifacts.put(
        schema_uri=resources.graph_schema_uri,
        semantics_uri=resources.semantics_uri,
        payload=graph_contract_from_value(graph).model_dump(mode="json"),
        parents=parents,
        summary=summary,
    )


def load_graph_value(
    resources: GraphArtifactResources,
    graph_uri: str,
) -> SimpleUndirectedGraph:
    """Load and validate one graph artifact as its immutable semantic value."""

    try:
        artifact = resources.store.get(graph_uri)
    except StorageError as exc:
        raise OperationInvocationError(
            OperationDiagnostic(
                code="GRAPH_ARTIFACT_NOT_FOUND",
                stage="graph_resolution",
                message="The requested graph artifact is unavailable.",
                path="graph_uri",
                hint=(
                    "Use a graph URI returned by graph.construct.explicit, "
                    "graph.search.atlas, or another graph-domain producer."
                ),
            )
        ) from exc
    if (
        artifact.manifest.schema_uri != resources.graph_schema_uri
        or artifact.manifest.semantics_uri != resources.semantics_uri
        or not isinstance(artifact.payload, dict)
    ):
        raise OperationInvocationError(
            OperationDiagnostic(
                code="INCOMPATIBLE_GRAPH_ARTIFACT",
                stage="graph_validation",
                message="The artifact is not a compatible simple undirected graph.",
                path="graph_uri",
                schema_uri=resources.graph_schema_uri,
                hint=(
                    "Use a graph URI returned by graph.construct.explicit, "
                    "graph.search.atlas, or another graph-domain producer."
                ),
            )
        )
    try:
        contract = SimpleUndirectedGraphContract.model_validate(artifact.payload)
        return graph_value_from_contract(contract)
    except ValidationError as exc:
        raise OperationInvocationError(
            OperationDiagnostic(
                code="INCOMPATIBLE_GRAPH_ARTIFACT",
                stage="graph_validation",
                message="The graph artifact payload is malformed.",
                path="graph_uri",
                schema_uri=resources.graph_schema_uri,
                hint="Recreate the graph through its owning operation.",
            )
        ) from exc


def load_graph(resources: GraphArtifactResources, graph_uri: str) -> nx_type.Graph[str]:
    """Load and validate one graph artifact against the installed graph contract."""
    payload = load_graph_value(resources, graph_uri)
    graph: nx_type.Graph[str] = nx().Graph()
    graph.add_nodes_from(payload.vertices)
    graph.add_edges_from(payload.edges)
    return graph


def runtime_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))


__all__ = [
    "ARTIFACT_URI_PATTERN",
    "GRAPH_PAYLOAD_SCHEMA",
    "GraphArtifactResources",
    "graph_payload",
    "load_graph",
    "load_graph_value",
    "nx",
    "publish_graph",
    "runtime_ms",
]
