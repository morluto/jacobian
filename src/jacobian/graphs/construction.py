"""Explicit graph construction capability."""

from __future__ import annotations

import time
from dataclasses import dataclass

from pydantic import ValidationError

from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInvocationExample,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.graph_composition import (
    GraphExplicitConstructionOutput,
    GraphExplicitConstructionRequest,
)
from jacobian.contracts.graph_isomorphism import SimpleUndirectedGraph
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.graphs.artifacts import GraphArtifactResources, runtime_ms
from jacobian.provider_runtime import known_provider_runtime
from jacobian.schema_registry import model_schema


@dataclass(frozen=True, slots=True)
class GraphConstructionResources:
    graph: GraphArtifactResources


class GraphExplicitConstructionAdapter:
    """Materialize one caller-supplied graph through the graph domain contract."""

    def __init__(self, resources: GraphConstructionResources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="graph.construct.explicit",
            version="1",
            title="Materialize an explicit simple graph",
            description=(
                "Validate and canonicalize one bounded explicit finite simple "
                "undirected graph, then return a domain-owned graph artifact accepted "
                "by graph capabilities."
            ),
            provider="jacobian.networkx",
            provider_runtime=known_provider_runtime(
                "jacobian.networkx",
                features=("simple-undirected-graphs", "canonical-materialization"),
            ),
            input_schema=model_schema(GraphExplicitConstructionRequest),
            output_schema=model_schema(GraphExplicitConstructionOutput),
            tags=("graph", "construction", "explicit", "artifact-materialization"),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="three-vertex-path",
                    description=(
                        "Materialize a path while allowing noncanonical caller order."
                    ),
                    input={
                        "vertices": ["c", "a", "b"],
                        "edges": [["b", "a"], ["c", "b"]],
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        started = time.monotonic()
        try:
            validated = GraphExplicitConstructionRequest.model_validate(request.input)
        except ValidationError as exc:
            raw_errors = exc.errors(include_url=False, include_context=False)
            errors = [dict(item) for item in raw_errors]
            path_parts = raw_errors[0]["loc"] if raw_errors else ()
            path = ".".join(str(item) for item in path_parts) or None
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_EXPLICIT_GRAPH",
                    stage="graph_input_validation",
                    message=(
                        "The complete explicit graph request violates finite "
                        "simple-undirected-graph semantics."
                    ),
                    path=path,
                    schema_uri=self.resources.graph.graph_schema_uri,
                    expected=(
                        "at most 256 unique nonempty string vertices and at most "
                        "32,640 unique non-loop edges over declared endpoints"
                    ),
                    hint=(
                        "Correct the reported vertices or edges; validation completes "
                        "before any graph artifact is written."
                    ),
                    details={"validation_errors": errors},
                )
            ) from exc

        vertices = tuple(sorted(validated.vertices))
        edges = tuple(
            sorted(
                (left, right) if left < right else (right, left)
                for left, right in validated.edges
            )
        )
        graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)
        graph_payload = graph.model_dump(mode="json")
        stored = self.resources.graph.artifacts.put(
            schema_uri=self.resources.graph.graph_schema_uri,
            semantics_uri=self.resources.graph.semantics_uri,
            payload=graph_payload,
            summary=(
                f"explicit simple undirected graph of order {len(vertices)} "
                f"and size {len(edges)}"
            ),
        )
        output = GraphExplicitConstructionOutput(
            graph_uri=stored.artifact_uri,
            graph_object_digest=stored.object_digest,
            graph_schema_uri=self.resources.graph.graph_schema_uri,
            graph_semantics_uri=self.resources.graph.semantics_uri,
            graph=graph,
            order=len(vertices),
            size=len(edges),
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=runtime_ms(started),
            ),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="the complete caller-supplied finite simple graph",
                parameters={
                    "order": len(vertices),
                    "size": len(edges),
                    "graph_schema_uri": self.resources.graph.graph_schema_uri,
                    "graph_semantics_uri": self.resources.graph.semantics_uri,
                },
                artifact_uri=stored.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.COMPLETE,
                basis=(
                    "the complete validated vertex and edge sets were canonically "
                    "materialized"
                ),
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis=(
                    "domain contract validation and deterministic canonicalization; "
                    "no mathematical property was asserted"
                ),
            ),
            artifact_uris=(stored.artifact_uri,),
        )
