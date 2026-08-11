"""Exact neighborhood-independence profile capability."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json, format_canonical_integer
from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInputKind,
    CapabilityRelationship,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.graph_invariants import (
    GraphNeighborhoodIndependenceArtifact,
    GraphNeighborhoodIndependenceClaim,
    GraphNeighborhoodIndependenceOutput,
    GraphNeighborhoodIndependenceRecord,
    GraphNeighborhoodIndependenceReplayPayload,
    GraphNeighborhoodIndependenceRequest,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.graphs.artifacts import GraphArtifactResources, load_graph, nx, runtime_ms
from jacobian.provider_runtime import known_provider_runtime
from jacobian.schema_registry import model_schema

if TYPE_CHECKING:
    import networkx as nx_type


@dataclass(frozen=True, slots=True)
class GraphNeighborhoodIndependenceResources:
    graph: GraphArtifactResources
    neighborhood_schema_uri: str
    neighborhood_claim_schema_uri: str
    certificate_schema_uri: str
    neighborhood_checker_id: str | None


class GraphNeighborhoodIndependenceAdapter:
    """Compute every exact neighborhood independence number for one graph."""

    def __init__(self, resources: GraphNeighborhoodIndependenceResources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="graph.compute.neighborhood_independence",
            version="1",
            title="Compute neighborhood independence",
            description=(
                "Compute an exact maximum independent set in every open "
                "neighborhood, their sum, and their rational average. "
                "Neighborhoods are limited to 24 vertices."
            ),
            provider="jacobian.networkx",
            provider_runtime=known_provider_runtime(
                "jacobian.networkx",
                features=("neighborhood-independence", "simple-undirected-graphs"),
                checker_ids=(
                    (resources.neighborhood_checker_id,)
                    if resources.neighborhood_checker_id is not None
                    else ()
                ),
            ),
            input_schema=model_schema(GraphNeighborhoodIndependenceRequest),
            output_schema=model_schema(GraphNeighborhoodIndependenceOutput),
            tags=(
                "graph",
                "neighborhood",
                "independence-number",
                "exact-computation",
            ),
            accepted_input_kinds=(CapabilityInputKind.TYPED_ARTIFACT,),
            accepted_artifact_types=(resources.graph.graph_schema_uri,),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = GraphNeighborhoodIndependenceRequest.model_validate(
                request.input
            )
        except ValidationError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_NEIGHBORHOOD_INDEPENDENCE_REQUEST",
                    stage="request_validation",
                    message=(
                        "The complete neighborhood-independence request is invalid."
                    ),
                )
            ) from exc
        started = time.monotonic()
        graph = load_graph(self.resources.graph, validated.graph_uri)
        if graph.number_of_nodes() > 256:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="GRAPH_ORDER_LIMIT_EXCEEDED",
                    stage="invariant_computation",
                    message=("The graph exceeds the exact 256-vertex profile limit."),
                )
            )
        records: list[GraphNeighborhoodIndependenceRecord] = []
        for vertex in sorted(graph):
            neighborhood = tuple(sorted(graph.neighbors(vertex)))
            if len(neighborhood) > 24:
                raise CapabilityInvocationError(
                    CapabilityDiagnostic(
                        code="NEIGHBORHOOD_ORDER_LIMIT_EXCEEDED",
                        stage="invariant_computation",
                        message=(
                            "At least one open neighborhood exceeds the exact "
                            "24-vertex operation limit."
                        ),
                        hint=(
                            "Use a structurally certified bound or a separately "
                            "budgeted solver-backed capability."
                        ),
                    )
                )
            neighborhood_graph: nx_type.Graph[str] = nx().Graph()
            neighborhood_graph.add_nodes_from(neighborhood)
            neighborhood_graph.add_edges_from(graph.subgraph(neighborhood).edges())
            independent_set, independence_number = nx().max_weight_clique(
                nx().complement(neighborhood_graph),
                weight=None,
            )
            records.append(
                GraphNeighborhoodIndependenceRecord(
                    vertex=vertex,
                    neighborhood=neighborhood,
                    independent_set=tuple(sorted(independent_set)),
                    independence_number=independence_number,
                )
            )
        total = sum(record.independence_number for record in records)
        average = Fraction(total, len(records)) if records else Fraction(0)
        average_wire = CanonicalRational(
            num=format_canonical_integer(average.numerator),
            den=format_canonical_integer(average.denominator),
        )
        invariant = GraphNeighborhoodIndependenceArtifact(
            graph_uri=validated.graph_uri,
            records=tuple(records),
            total=total,
            average=average_wire,
            backend_version=nx().__version__,
        )
        invariant_artifact = self.resources.graph.artifacts.put(
            schema_uri=self.resources.neighborhood_schema_uri,
            semantics_uri=self.resources.graph.semantics_uri,
            payload=invariant.model_dump(mode="json"),
            parents=(validated.graph_uri,),
            summary="exact graph neighborhood-independence profile",
        )
        claim = GraphNeighborhoodIndependenceClaim(source_graph_uri=validated.graph_uri)
        claim_artifact = self.resources.graph.artifacts.put(
            schema_uri=self.resources.neighborhood_claim_schema_uri,
            semantics_uri=self.resources.graph.semantics_uri,
            payload=claim.model_dump(mode="json"),
            parents=(validated.graph_uri, invariant_artifact.artifact_uri),
            summary="exact neighborhood-independence profile claim",
        )
        semantics = self.resources.graph.store.get(self.resources.graph.semantics_uri)
        source_graph = self.resources.graph.store.get(validated.graph_uri)
        certificate_payload = GraphNeighborhoodIndependenceReplayPayload(
            source_graph_uri=validated.graph_uri,
            invariant_uri=invariant_artifact.artifact_uri,
        ).model_dump(mode="json")
        certificate = CertificateEnvelope(
            certificate_type="graph.neighborhood_independence",
            format_version="1",
            bindings=EvidenceBindings(
                claim_digest=claim_artifact.object_digest,
                semantics_digest=semantics.manifest.object_digest,
                candidate_digest=invariant_artifact.object_digest,
                scope_digest=source_graph.manifest.object_digest,
            ),
            payload_digest=(
                "sha256:"
                + hashlib.sha256(canonicalize_json(certificate_payload)).hexdigest()
            ),
            payload=certificate_payload,
        )
        certificate_artifact = self.resources.graph.artifacts.put(
            schema_uri=self.resources.certificate_schema_uri,
            semantics_uri=self.resources.graph.semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=(
                claim_artifact.artifact_uri,
                invariant_artifact.artifact_uri,
                validated.graph_uri,
            ),
            summary="unverified graph neighborhood-independence certificate",
        )
        output = GraphNeighborhoodIndependenceOutput(
            graph_uri=validated.graph_uri,
            invariant_uri=invariant_artifact.artifact_uri,
            claim_uri=claim_artifact.artifact_uri,
            certificate_uri=certificate_artifact.artifact_uri,
            checker_id=self.resources.neighborhood_checker_id,
            records=tuple(records),
            total=total,
            average=average_wire,
            backend_version=nx().__version__,
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
                description=(
                    "all open neighborhoods of one finite simple undirected graph"
                ),
                parameters={
                    "graph_uri": validated.graph_uri,
                    "graph_order": graph.number_of_nodes(),
                    "maximum_neighborhood_order": 24,
                },
                artifact_uri=validated.graph_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.COMPLETE,
                basis=(
                    "every open neighborhood was solved exactly within the "
                    "advertised 24-vertex limit; verification remains separate"
                ),
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            relationships=(
                CapabilityRelationship(
                    relation_id=("graph.relation.neighborhood-independence-profile-of"),
                    source_artifact_uris=(validated.graph_uri,),
                    target_artifact_uris=(invariant_artifact.artifact_uri,),
                ),
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis=(
                    "NetworkX exact maximum-clique computations on complement "
                    "neighborhoods; the bundled certificate was not invoked"
                ),
            ),
            artifact_uris=(
                validated.graph_uri,
                invariant_artifact.artifact_uri,
                claim_artifact.artifact_uri,
                certificate_artifact.artifact_uri,
            ),
        )
