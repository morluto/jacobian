"""Fixed-registry graph invariant computation capability."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from jacobian.capabilities import CapabilityInvocationError
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
from jacobian.contracts.graph_invariants import (
    GraphInvariantBatchArtifact,
    GraphInvariantBatchOutput,
    GraphInvariantBatchRequest,
    GraphInvariantBinding,
    GraphInvariantResult,
    GraphInvariantResultArtifact,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.graphs.artifacts import (
    GraphArtifactResources,
    load_graph,
    nx,
    runtime_ms,
)
from jacobian.provider_runtime import known_provider_runtime
from jacobian.schema_registry import model_schema

if TYPE_CHECKING:
    import networkx as nx_type


PROPERTY_NAMES = (
    "average_eccentricity",
    "bipartite",
    "connected",
    "degree_sequence",
    "diameter",
    "eccentricities",
    "girth",
    "harmonic_index",
    "havel_hakimi_trace",
    "independence_number",
    "maximum_degree",
    "minimum_degree",
    "order",
    "radius",
    "residue",
    "size",
    "tree",
    "triangle_count",
    "triangle_frequencies",
)

MAX_EXACT_INDEPENDENCE_ORDER = 24


@dataclass(frozen=True, slots=True)
class GraphInvariantResources:
    graph: GraphArtifactResources
    property_schema_uri: str
    invariant_result_schema_uri: str


class GraphPropertyAdapter:
    """Compute a requested batch of exact properties for one graph artifact."""

    def __init__(self, resources: GraphInvariantResources) -> None:
        self.resources = resources
        input_schema = model_schema(GraphInvariantBatchRequest)
        input_schema["x-supported-invariants"] = list(PROPERTY_NAMES)
        self._descriptor = CapabilityDescriptor(
            capability_id="graph.compute.properties",
            version="2",
            title="Compute exact graph properties",
            description=(
                "Classify and compute a requested batch against the fixed exact "
                "graph-invariant registry, preserving every per-invariant outcome."
            ),
            provider="jacobian.networkx",
            provider_runtime=known_provider_runtime(
                "jacobian.networkx",
                features=("graph-properties", "simple-undirected-graphs"),
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=input_schema,
            output_schema=model_schema(GraphInvariantBatchOutput),
            tags=("graph", "properties", "exact-computation"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        started = time.monotonic()
        try:
            validated = GraphInvariantBatchRequest.model_validate(request.input)
        except ValidationError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_GRAPH_INVARIANT_BATCH_REQUEST",
                    stage="request_validation",
                    message=(
                        "The complete graph-invariant batch request is invalid: "
                        f"{exc.errors()[0].get('msg', 'validation failed')}"
                    ),
                    hint=(
                        "Supply one compatible graph artifact URI and 1 to 32 "
                        "unique lowercase invariant names."
                    ),
                )
            ) from exc
        graph_uri = validated.graph_uri
        graph = load_graph(self.resources.graph, graph_uri)
        names = tuple(sorted(validated.properties))
        bindings: list[GraphInvariantBinding] = []
        selected: dict[str, Any] = {}
        for name in names:
            result = _compute_invariant_result(graph, name)
            if result.status == "COMPUTED":
                selected[name] = {
                    "value": result.value,
                    "exactness": result.exactness,
                    "backend": result.backend,
                }
            result_artifact = self.resources.graph.artifacts.put(
                schema_uri=self.resources.invariant_result_schema_uri,
                semantics_uri=self.resources.graph.semantics_uri,
                payload=GraphInvariantResultArtifact(
                    graph_uri=graph_uri,
                    backend_version=nx().__version__,
                    result=result,
                ).model_dump(mode="json"),
                parents=(graph_uri,),
                summary=f"{result.status.lower()} graph invariant: {name}",
            )
            bindings.append(
                GraphInvariantBinding(
                    invariant=name,
                    artifact_uri=result_artifact.artifact_uri,
                    result=result,
                )
            )
        batch = GraphInvariantBatchArtifact(
            graph_uri=graph_uri,
            supported_invariants=tuple(sorted(PROPERTY_NAMES)),
            requested_invariants=names,
            backend_version=nx().__version__,
            results=tuple(bindings),
            properties=selected,
        )
        property_artifact = self.resources.graph.artifacts.put(
            schema_uri=self.resources.property_schema_uri,
            semantics_uri=self.resources.graph.semantics_uri,
            payload=batch.model_dump(mode="json"),
            parents=(graph_uri, *(binding.artifact_uri for binding in bindings)),
            summary="fixed-registry graph-invariant batch",
        )
        output = GraphInvariantBatchOutput(
            **batch.model_dump(mode="python"),
            property_artifact_uri=property_artifact.artifact_uri,
        )
        has_incomplete_results = any(
            binding.result.status == "NOT_COMPUTED" for binding in bindings
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=runtime_ms(started),
            ),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="the requested property batch for one exact graph artifact",
                parameters={
                    "graph_uri": graph_uri,
                    "registry_version": "1",
                    "properties": list(names),
                },
                artifact_uri=graph_uri,
            ),
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.PARTIAL
                    if has_incomplete_results
                    else CapabilityCompletenessStatus.COMPLETE
                ),
                basis=(
                    "at least one requested invariant was not computed because "
                    "the declared exact scope exceeded its safety boundary"
                    if has_incomplete_results
                    else "every requested invariant received a terminal COMPUTED, "
                    "NOT_APPLICABLE, or UNSUPPORTED result under registry version 1"
                ),
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            relationships=(
                CapabilityRelationship(
                    relation_id="graph.relation.properties-of",
                    source_artifact_uris=(graph_uri,),
                    target_artifact_uris=(
                        property_artifact.artifact_uri,
                        *(binding.artifact_uri for binding in bindings),
                    ),
                ),
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis=(
                    "deterministic exact NetworkX algorithms; no independent "
                    "checker was invoked"
                ),
            ),
            artifact_uris=(
                graph_uri,
                property_artifact.artifact_uri,
                *(binding.artifact_uri for binding in bindings),
            ),
        )


def _compute_property(graph: nx_type.Graph[Any], name: str) -> Any:
    """Compute only the requested outcome instead of the full property portfolio."""

    if name == "order":
        return graph.number_of_nodes()
    if name == "size":
        return graph.number_of_edges()
    if name == "connected":
        return nx().is_connected(graph) if graph else False
    if name == "bipartite":
        return nx().is_bipartite(graph)
    if name == "tree":
        return nx().is_tree(graph) if graph else False
    if name in {"degree_sequence", "minimum_degree", "maximum_degree"}:
        degrees = sorted((degree for _, degree in graph.degree), reverse=True)
        if name == "degree_sequence":
            return degrees
        if name == "minimum_degree":
            return min(degrees) if degrees else None
        return max(degrees) if degrees else None
    if name == "triangle_count":
        return sum(cast(dict[Any, int], nx().triangles(graph)).values()) // 3
    if name == "independence_number":
        if not graph:
            return 0
        independent_set, independence_number = nx().max_weight_clique(
            nx().complement(graph),
            weight=None,
        )
        assert len(independent_set) == independence_number
        return independence_number
    if name == "girth":
        value = nx().girth(graph)
        return None if math.isinf(value) else int(value)
    if name in {"eccentricities", "diameter", "radius", "average_eccentricity"}:
        if not graph:
            raise nx().NetworkXPointlessConcept(
                "distance properties are undefined for the null graph"
            )
        eccentricities = cast(dict[Any, int], nx().eccentricity(graph))
        ordered = {
            str(vertex): eccentricities[vertex]
            for vertex in sorted(eccentricities, key=str)
        }
        if name == "eccentricities":
            return ordered
        if name == "diameter":
            return max(eccentricities.values(), default=0)
        if name == "radius":
            return min(eccentricities.values(), default=0)
        total = sum(eccentricities.values())
        return _rational_payload(Fraction(total, len(eccentricities)))
    if name == "triangle_frequencies":
        frequencies = cast(dict[Any, int], nx().triangles(graph))
        return {
            str(vertex): frequencies[vertex] for vertex in sorted(frequencies, key=str)
        }
    if name == "harmonic_index":
        harmonic_value = Fraction(0)
        for source, target in graph.edges:
            harmonic_value += Fraction(
                2,
                graph.degree[source] + graph.degree[target],
            )
        return _rational_payload(harmonic_value)
    if name in {"havel_hakimi_trace", "residue"}:
        trace = _havel_hakimi_trace(graph)
        return trace if name == "havel_hakimi_trace" else len(trace[-1])
    raise AssertionError(f"unsupported graph property: {name}")


def _havel_hakimi_trace(graph: nx_type.Graph[Any]) -> list[list[int]]:
    sequence = sorted((degree for _, degree in graph.degree), reverse=True)
    trace = [sequence.copy()]
    while sequence and sequence[0] > 0:
        degree = sequence.pop(0)
        if degree > len(sequence):
            raise nx().NetworkXError("degree sequence became non-graphical")
        for index in range(degree):
            sequence[index] -= 1
            if sequence[index] < 0:
                raise nx().NetworkXError("degree sequence became non-graphical")
        sequence.sort(reverse=True)
        trace.append(sequence.copy())
    return trace


def _rational_payload(value: Fraction) -> dict[str, str]:
    return {"num": str(value.numerator), "den": str(value.denominator)}


def _property_backend(name: str) -> str:
    backend = (
        "networkx.max_weight_clique(complement)"
        if name == "independence_number"
        else "networkx"
    )
    return backend


def _compute_invariant_result(
    graph: nx_type.Graph[Any],
    name: str,
) -> GraphInvariantResult:
    if name not in PROPERTY_NAMES:
        return GraphInvariantResult(
            invariant=name,
            status="UNSUPPORTED",
            exactness="NOT_APPLICABLE",
            detail=(
                "the invariant is not present in graph.compute.properties "
                "registry version 1"
            ),
        )
    if name == "independence_number" and graph.number_of_nodes() > (
        MAX_EXACT_INDEPENDENCE_ORDER
    ):
        return GraphInvariantResult(
            invariant=name,
            status="NOT_COMPUTED",
            exactness="NOT_APPLICABLE",
            backend=_property_backend(name),
            detail=(
                "exact independence-number computation is limited to graphs of "
                f"order {MAX_EXACT_INDEPENDENCE_ORDER}; received order "
                f"{graph.number_of_nodes()}"
            ),
        )
    try:
        value = _compute_property(graph, name)
    except cast(type[BaseException], nx().NetworkXError) as exc:
        return GraphInvariantResult(
            invariant=name,
            status="NOT_APPLICABLE",
            exactness="NOT_APPLICABLE",
            backend=_property_backend(name),
            detail=str(exc),
        )
    return GraphInvariantResult(
        invariant=name,
        status="COMPUTED",
        value=value,
        exactness="EXACT",
        backend=_property_backend(name),
    )
