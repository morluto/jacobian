"""Explicit graph construction capability."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from jacobian.capability_errors import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInvocationExample,
    CapabilityRequest,
)
from jacobian.contracts.graph_composition import (
    GraphExplicitConstructionOutput,
    GraphExplicitConstructionRequest,
)
from jacobian.graphs.artifacts import (
    GraphArtifactResources,
    publish_graph,
)
from jacobian.graphs.conversions import graph_contract_from_value
from jacobian.math.graphs import SimpleUndirectedGraph, explicit_graph
from jacobian.operation_execution import execute_operation
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed, OperationSpec
from jacobian.provider_runtime import known_provider_runtime
from jacobian.schema_registry import model_schema


@dataclass(frozen=True, slots=True)
class GraphConstructionResources:
    graph: GraphArtifactResources


class GraphExplicitConstructionAdapter:
    """Materialize one caller-supplied graph through the graph domain contract."""

    def __init__(self, resources: GraphConstructionResources) -> None:
        self.resources = resources
        self.spec = OperationSpec(
            operation_id="graph.construct.explicit",
            version="1",
            request_type=GraphExplicitConstructionRequest,
            result_type=SimpleUndirectedGraph,
            execute=_explicit_graph,
            title="Materialize an explicit simple graph",
            description=(
                "Validate and canonicalize one bounded explicit finite simple "
                "undirected graph, then return a domain-owned graph artifact accepted "
                "by graph capabilities."
            ),
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
        self._descriptor = CapabilityDescriptor(
            capability_id=self.spec.operation_id,
            version=self.spec.version,
            title=self.spec.title,
            description=self.spec.description,
            provider="jacobian.networkx",
            provider_runtime=known_provider_runtime(
                "jacobian.networkx",
                features=("simple-undirected-graphs", "canonical-materialization"),
            ),
            input_schema=model_schema(GraphExplicitConstructionRequest),
            output_schema=model_schema(GraphExplicitConstructionOutput),
            tags=self.spec.tags,
            invocation_examples=self.spec.invocation_examples,
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> OperationProjection:
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

        terminal = execute_operation(self.spec, validated)
        if not isinstance(terminal, Completed):
            return OperationProjection(
                operation_id=self.spec.operation_id,
                version=self.spec.version,
                terminal=terminal,
            )
        graph = terminal.value
        stored = publish_graph(
            self.resources.graph,
            graph,
            summary=(
                f"explicit simple undirected graph of order {len(graph.vertices)} "
                f"and size {len(graph.edges)}"
            ),
        )
        output = GraphExplicitConstructionOutput(
            graph_uri=stored.artifact_uri,
            graph_object_digest=stored.object_digest,
            graph_schema_uri=self.resources.graph.graph_schema_uri,
            graph_semantics_uri=self.resources.graph.semantics_uri,
            graph=graph_contract_from_value(graph),
            order=len(graph.vertices),
            size=len(graph.edges),
        )
        return OperationProjection(
            operation_id=self.spec.operation_id,
            version=self.spec.version,
            terminal=terminal,
            publication=PublishedOperation(
                output=output,
                artifact_uris=(stored.artifact_uri,),
            ),
        )


def _explicit_graph(
    request: GraphExplicitConstructionRequest,
) -> SimpleUndirectedGraph:
    """Bind the operation request to the graph library's semantic input."""

    return explicit_graph(request.vertices, request.edges)
