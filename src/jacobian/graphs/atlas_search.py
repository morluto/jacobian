"""Graph Atlas search capability."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityMode,
    CapabilityRelationship,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.domains._examples import example
from jacobian.graphs.artifacts import (
    ARTIFACT_URI_PATTERN,
    GRAPH_PAYLOAD_SCHEMA,
    GraphArtifactResources,
    nx,
    runtime_ms,
)
from jacobian.graphs.artifacts import (
    graph_payload as canonical_graph_payload,
)
from jacobian.graphs.atlas import graph_atlas_order
from jacobian.provider_runtime import known_provider_runtime

if TYPE_CHECKING:
    import networkx as nx_type


_CONSTRAINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "connected": {"type": "boolean"},
        "bipartite": {"type": "boolean"},
        "tree": {"type": "boolean"},
        "triangle_free": {"type": "boolean"},
        "minimum_edges": {"type": "integer", "minimum": 0},
        "maximum_edges": {"type": "integer", "minimum": 0},
        "minimum_degree": {"type": "integer", "minimum": 0},
        "maximum_degree": {"type": "integer", "minimum": 0},
        "independence_number": {"type": "integer", "minimum": 0},
    },
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class GraphAtlasSearchResources:
    graph: GraphArtifactResources
    scope_schema_uri: str


class GraphAtlasSearchAdapter:
    """Search NetworkX's bounded Graph Atlas using exact computed properties."""

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
            modes=(CapabilityMode.EXPLORE,),
            input_schema={
                "type": "object",
                "properties": {
                    "order": {"type": "integer", "minimum": 0, "maximum": 7},
                    "constraints": _CONSTRAINT_SCHEMA,
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["order", "constraints"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "graph_uri": {
                                    "type": "string",
                                    "pattern": ARTIFACT_URI_PATTERN,
                                },
                                "graph": GRAPH_PAYLOAD_SCHEMA,
                                "properties": {"type": "object"},
                            },
                            "required": ["graph_uri", "graph", "properties"],
                            "additionalProperties": False,
                        },
                    },
                    "match_count": {"type": "integer", "minimum": 0},
                    "returned_count": {"type": "integer", "minimum": 0},
                    "truncated": {"type": "boolean"},
                    "scope_uri": {
                        "type": "string",
                        "pattern": ARTIFACT_URI_PATTERN,
                    },
                    "backend": {"const": "networkx.graph_atlas_g"},
                    "backend_version": {"type": "string"},
                },
                "required": [
                    "candidates",
                    "match_count",
                    "returned_count",
                    "truncated",
                    "scope_uri",
                    "backend",
                    "backend_version",
                ],
                "additionalProperties": False,
            },
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

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        started = time.monotonic()
        order = int(request.input["order"])
        constraints = dict(request.input["constraints"])
        _validate_constraint_ranges(constraints)
        limit = int(request.input.get("limit", 10))
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
        matches: list[tuple[nx_type.Graph[Any], dict[str, Any]]] = []
        for graph in atlas_graphs:
            properties = _compute_all_properties(graph)
            if _matches_constraints(properties, constraints):
                matches.append((graph, properties))
        candidates: list[dict[str, Any]] = []
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
                {
                    "graph_uri": graph_artifact.artifact_uri,
                    "graph": graph_payload,
                    "properties": properties,
                }
            )
        artifact_uris = (scope.artifact_uri, *graph_uris)
        relationships = tuple(
            CapabilityRelationship(
                relation_id="graph.relation.atlas-member",
                source_artifact_uris=(scope.artifact_uri,),
                target_artifact_uris=(graph_uri,),
            )
            for graph_uri in graph_uris
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=runtime_ms(started),
            ),
            output={
                "candidates": candidates,
                "match_count": len(matches),
                "returned_count": len(candidates),
                "truncated": len(matches) > len(candidates),
                "scope_uri": scope.artifact_uri,
                "backend": "networkx.graph_atlas_g",
                "backend_version": nx().__version__,
            },
            scope=CapabilityScope(
                description=(
                    "all Graph Atlas representatives with the requested exact order"
                ),
                parameters={
                    "source": "networkx.graph_atlas_g",
                    "backend_version": nx().__version__,
                    "order": order,
                },
                artifact_uri=scope.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.COMPLETE,
                basis=(
                    "the maintained Graph Atlas provider was scanned to exhaustion; "
                    "this computation was not independently checked"
                ),
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            relationships=relationships,
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis=(
                    "deterministic NetworkX Graph Atlas enumeration and exact "
                    "property filters; no independent checker was invoked"
                ),
            ),
            artifact_uris=artifact_uris,
        )


def _validate_constraint_ranges(constraints: dict[str, Any]) -> None:
    for lower, upper in (
        ("minimum_edges", "maximum_edges"),
        ("minimum_degree", "maximum_degree"),
    ):
        if (
            lower in constraints
            and upper in constraints
            and int(constraints[lower]) > int(constraints[upper])
        ):
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_CONSTRAINT_RANGE",
                    stage="constraint_validation",
                    message=f"{lower} cannot exceed {upper}.",
                    path=f"constraints/{lower}",
                    hint="Swap the bounds or remove one of them, then retry.",
                )
            )


def _compute_all_properties(graph: nx_type.Graph[Any]) -> dict[str, Any]:
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
    return {
        "order": order,
        "size": graph.number_of_edges(),
        "connected": nx().is_connected(graph) if order else False,
        "bipartite": nx().is_bipartite(graph),
        "tree": nx().is_tree(graph) if order else False,
        "degree_sequence": degrees,
        "minimum_degree": min(degrees) if degrees else None,
        "maximum_degree": max(degrees) if degrees else None,
        "triangle_count": (
            sum(cast(dict[Any, int], nx().triangles(graph)).values()) // 3
        ),
        "independence_number": independence_number,
    }


def _matches_constraints(
    properties: dict[str, Any],
    constraints: dict[str, Any],
) -> bool:
    if (
        "connected" in constraints
        and properties["connected"] is not constraints["connected"]
    ):
        return False
    if (
        "bipartite" in constraints
        and properties["bipartite"] is not constraints["bipartite"]
    ):
        return False
    if "tree" in constraints and properties["tree"] is not constraints["tree"]:
        return False
    if "triangle_free" in constraints and (
        (properties["triangle_count"] == 0) is not constraints["triangle_free"]
    ):
        return False
    if (
        "minimum_edges" in constraints
        and properties["size"] < constraints["minimum_edges"]
    ):
        return False
    if (
        "maximum_edges" in constraints
        and properties["size"] > constraints["maximum_edges"]
    ):
        return False
    if "minimum_degree" in constraints and (
        properties["minimum_degree"] is None
        or properties["minimum_degree"] < constraints["minimum_degree"]
    ):
        return False
    if "maximum_degree" in constraints and (
        properties["maximum_degree"] is None
        or properties["maximum_degree"] > constraints["maximum_degree"]
    ):
        return False
    return not (
        "independence_number" in constraints
        and properties["independence_number"] != constraints["independence_number"]
    )
