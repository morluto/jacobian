"""Graph Atlas search capability."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from jacobian.capability_errors import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityRequest,
)
from jacobian.contracts.graph_invariants import (
    GraphAtlasCandidate,
    GraphAtlasConstraints,
    GraphAtlasProperties,
    GraphAtlasSearchOutput,
    GraphAtlasSearchRequest,
)
from jacobian.domains._examples import example
from jacobian.graphs.artifacts import (
    GraphArtifactResources,
    nx,
    runtime_ms,
)
from jacobian.graphs.artifacts import (
    graph_payload as canonical_graph_payload,
)
from jacobian.graphs.atlas import graph_atlas_order
from jacobian.graphs.conversions import graph_contract_from_value
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed
from jacobian.provider_runtime import known_provider_runtime
from jacobian.schema_registry import model_schema

if TYPE_CHECKING:
    import networkx as nx_type


@dataclass(frozen=True, slots=True)
class GraphAtlasSearchResources:
    graph: GraphArtifactResources
    scope_schema_uri: str


class GraphAtlasSearchAdapter:
    """Search NetworkX's bounded Graph Atlas using exact computed properties."""

    typed_input = True

    def __init__(self, resources: GraphAtlasSearchResources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="graph.search.atlas",
            version="1",
            title="Search the Graph Atlas",
            description=(
                "Search all Graph Atlas representatives of one exact order "
                "(0-7) using exact NetworkX-computed constraints."
            ),
            provider="jacobian.networkx",
            provider_runtime=known_provider_runtime(
                "jacobian.networkx",
                features=("graph-atlas", "simple-undirected-graphs"),
            ),
            input_schema=model_schema(GraphAtlasSearchRequest),
            output_schema=model_schema(GraphAtlasSearchOutput),
            tags=("graph", "construction", "bounded-search"),
            invocation_examples=(
                example(
                    "empty_graph_search",
                    "Find the order-zero graph in the atlas.",
                    {"order": 0, "constraints": {}, "limit": 1},
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> OperationProjection:
        started = time.monotonic()
        try:
            validated = GraphAtlasSearchRequest.model_validate(request.input)
        except ValidationError as exc:
            error = exc.errors()[0]
            error_message = str(error.get("msg", ""))
            invalid_range = "cannot exceed maximum_" in error_message
            range_path = (
                "constraints/minimum_degree"
                if "minimum_degree" in error_message
                else "constraints/minimum_edges"
            )
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code=(
                        "INVALID_CONSTRAINT_RANGE"
                        if invalid_range
                        else "INVALID_GRAPH_ATLAS_SEARCH_REQUEST"
                    ),
                    stage=(
                        "constraint_validation"
                        if invalid_range
                        else "request_validation"
                    ),
                    message=(
                        "The complete Graph Atlas search request is invalid: "
                        f"{error.get('msg', 'validation failed')}"
                    ),
                    path=range_path if invalid_range else None,
                    hint=(
                        "Swap the bounds or remove one of them, then retry."
                        if invalid_range
                        else (
                            "Supply an order from 0 to 7 and consistent exact "
                            "constraints."
                        )
                    ),
                )
            ) from exc
        order = validated.order
        constraints = validated.constraints
        limit = validated.limit
        atlas_graphs = graph_atlas_order(order)
        scope = self.resources.graph.artifacts.put(
            schema_uri=self.resources.scope_schema_uri,
            semantics_uri=self.resources.graph.semantics_uri,
            payload={
                "scope_schema_version": "1",
                "source": "networkx.graph_atlas_g",
                "backend_version": nx().__version__,
                "order": order,
                "enumerated_count": len(atlas_graphs),
            },
            summary=f"Graph Atlas representatives of order {order}",
        )
        matches: list[tuple[nx_type.Graph[Any], GraphAtlasProperties]] = []
        for graph in atlas_graphs:
            properties = _compute_all_properties(graph)
            if _matches_constraints(properties, constraints):
                matches.append((graph, properties))
        candidates: list[GraphAtlasCandidate] = []
        graph_uris: list[str] = []
        for graph, properties in matches[:limit]:
            graph_payload = canonical_graph_payload(graph)
            graph_artifact = self.resources.graph.artifacts.put(
                schema_uri=self.resources.graph.graph_schema_uri,
                semantics_uri=self.resources.graph.semantics_uri,
                payload=graph_payload,
                parents=(scope.artifact_uri,),
                summary=f"Graph Atlas candidate of order {order}",
            )
            graph_uris.append(graph_artifact.artifact_uri)
            candidates.append(
                GraphAtlasCandidate(
                    graph_uri=graph_artifact.artifact_uri,
                    graph=graph_contract_from_value(
                        SimpleUndirectedGraph.model_validate(graph_payload)
                    ),
                    properties=properties,
                )
            )
        artifact_uris = (scope.artifact_uri, *graph_uris)
        output = GraphAtlasSearchOutput(
            candidates=tuple(candidates),
            match_count=len(matches),
            returned_count=len(candidates),
            truncated=len(matches) > len(candidates),
            scope_uri=scope.artifact_uri,
            backend_version=nx().__version__,
        )
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=Completed(
                value=output,
                runtime_ms=runtime_ms(started),
            ),
            publication=PublishedOperation(output=output, artifact_uris=artifact_uris),
        )


def _compute_all_properties(graph: nx_type.Graph[Any]) -> GraphAtlasProperties:
    order = graph.number_of_nodes()
    degrees = sorted((degree for _, degree in graph.degree), reverse=True)
    if order:
        independent_set, independence_number = nx().max_weight_clique(
            nx().complement(graph),
            weight=None,
        )
        if len(independent_set) != independence_number:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INCONSISTENT_INDEPENDENCE_RESULT",
                    stage="backend_execution",
                    message=(
                        "The graph backend returned an independent-set witness whose "
                        "size does not match its reported independence number."
                    ),
                    hint="Retry with a supported graph backend.",
                )
            )
    else:
        independence_number = 0
    return GraphAtlasProperties(
        order=order,
        size=graph.number_of_edges(),
        connected=nx().is_connected(graph) if order else False,
        bipartite=nx().is_bipartite(graph),
        tree=nx().is_tree(graph) if order else False,
        degree_sequence=tuple(degrees),
        minimum_degree=min(degrees) if degrees else None,
        maximum_degree=max(degrees) if degrees else None,
        triangle_count=(sum(cast(dict[Any, int], nx().triangles(graph)).values()) // 3),
        independence_number=independence_number,
    )


def _matches_constraints(
    properties: GraphAtlasProperties,
    constraints: GraphAtlasConstraints,
) -> bool:
    if (
        constraints.connected is not None
        and properties.connected != constraints.connected
    ):
        return False
    if (
        constraints.bipartite is not None
        and properties.bipartite != constraints.bipartite
    ):
        return False
    if constraints.tree is not None and properties.tree != constraints.tree:
        return False
    if constraints.triangle_free is not None and (
        (properties.triangle_count == 0) != constraints.triangle_free
    ):
        return False
    if (
        constraints.minimum_edges is not None
        and properties.size < constraints.minimum_edges
    ):
        return False
    if (
        constraints.maximum_edges is not None
        and properties.size > constraints.maximum_edges
    ):
        return False
    if constraints.minimum_degree is not None and (
        properties.minimum_degree is None
        or properties.minimum_degree < constraints.minimum_degree
    ):
        return False
    if constraints.maximum_degree is not None and (
        properties.maximum_degree is None
        or properties.maximum_degree > constraints.maximum_degree
    ):
        return False
    return not (
        constraints.independence_number is not None
        and properties.independence_number != constraints.independence_number
    )
