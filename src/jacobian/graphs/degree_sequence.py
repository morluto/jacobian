"""Exact simple-graph degree-sequence realization operation."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.graph_degree_sequence import (
    GraphDegreeSequenceClaim,
    GraphDegreeSequenceObstruction,
    GraphDegreeSequenceOutput,
    GraphDegreeSequenceReplayPayload,
    GraphDegreeSequenceRequest,
    GraphDegreeSequenceResultArtifact,
)
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiagnostic,
    OperationRequest,
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
from jacobian.operation_adapters import parse_operation_input
from jacobian.operation_errors import OperationInvocationError
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed
from jacobian.provider_runtime import known_provider_runtime


@dataclass(frozen=True, slots=True)
class GraphDegreeSequenceResources:
    graph: GraphArtifactResources
    degree_sequence_claim_schema_uri: str
    degree_sequence_result_schema_uri: str
    certificate_schema_uri: str
    degree_sequence_checker_id: str | None


class GraphDegreeSequenceAdapter:
    """Realize a simple-graph degree sequence or expose an exact obstruction."""

    def __init__(self, resources: GraphDegreeSequenceResources) -> None:
        self.resources = resources
        self._descriptor = OperationDescriptor(
            operation_id="graph.realize.degree_sequence",
            version="1",
            title="Realize a simple-graph degree sequence",
            description=(
                "Construct a simple graph with the requested degree multiset, or "
                "return an odd-sum, maximum-degree, or Erdos-Gallai obstruction."
            ),
            provider="jacobian.networkx",
            provider_runtime=known_provider_runtime(
                "jacobian.networkx",
                features=("degree-sequence", "simple-undirected-graphs"),
                checker_ids=(
                    (resources.degree_sequence_checker_id,)
                    if resources.degree_sequence_checker_id is not None
                    else ()
                ),
            ),
            input_schema=GraphDegreeSequenceRequest.model_json_schema(),
            output_schema=GraphDegreeSequenceOutput.model_json_schema(),
            tags=(
                "graph",
                "degree-sequence",
                "construction",
                "counterexample",
            ),
            examples=(
                example(
                    "triangle_degree_sequence",
                    "Realize the degree sequence of a triangle.",
                    {"degree_sequence": [2, 2, 2]},
                ),
            ),
        )

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> GraphDegreeSequenceRequest:
        try:
            return parse_operation_input(GraphDegreeSequenceRequest, request.input)
        except ValidationError as exc:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="INVALID_DEGREE_SEQUENCE_REQUEST",
                    stage="request_validation",
                    message="The complete degree-sequence request is invalid.",
                    hint=("Provide between 1 and 512 nonnegative integer degrees."),
                )
            ) from exc

    def invoke(self, validated: GraphDegreeSequenceRequest) -> OperationProjection:
        started = time.monotonic()
        sequence = tuple(validated.degree_sequence)
        obstruction = _degree_sequence_obstruction(sequence)
        graph_payload: dict[str, Any] | None = None
        graph_uri: str | None = None
        graph_artifact = None
        conclusion: Literal["GRAPHICAL", "NON_GRAPHICAL"]
        method: Literal[
            "EXACT_DEGREE_REPLAY",
            "ODD_SUM_OBSTRUCTION",
            "MAX_DEGREE_OBSTRUCTION",
            "ERDOS_GALLAI_OBSTRUCTION",
        ]
        if obstruction is None:
            graph = nx().havel_hakimi_graph(sequence)
            graph_payload = canonical_graph_payload(graph)
            graph_artifact = self.resources.graph.artifacts.put(
                schema_uri=self.resources.graph.graph_schema_uri,
                semantics_uri=self.resources.graph.semantics_uri,
                payload=graph_payload,
                summary="simple graph realizing the requested degree sequence",
            )
            graph_uri = graph_artifact.artifact_uri
            conclusion = "GRAPHICAL"
            method = "EXACT_DEGREE_REPLAY"
        else:
            conclusion = "NON_GRAPHICAL"
            method = cast(
                Literal[
                    "ODD_SUM_OBSTRUCTION",
                    "MAX_DEGREE_OBSTRUCTION",
                    "ERDOS_GALLAI_OBSTRUCTION",
                ],
                {
                    "ODD_SUM": "ODD_SUM_OBSTRUCTION",
                    "MAX_DEGREE": "MAX_DEGREE_OBSTRUCTION",
                    "ERDOS_GALLAI": "ERDOS_GALLAI_OBSTRUCTION",
                }[obstruction.kind],
            )
        claim = GraphDegreeSequenceClaim(degree_sequence=sequence)
        claim_artifact = self.resources.graph.artifacts.put(
            schema_uri=self.resources.degree_sequence_claim_schema_uri,
            semantics_uri=self.resources.graph.semantics_uri,
            payload=claim.model_dump(mode="json"),
            summary="simple-graph degree-sequence realizability claim",
        )
        result_artifact_payload = GraphDegreeSequenceResultArtifact(
            degree_sequence=sequence,
            conclusion=conclusion,
            graph_uri=graph_uri,
            graph=graph_payload,
            obstruction=obstruction,
        )
        result_parents = (
            (claim_artifact.artifact_uri, graph_uri)
            if graph_uri is not None
            else (claim_artifact.artifact_uri,)
        )
        result_artifact = self.resources.graph.artifacts.put(
            schema_uri=self.resources.degree_sequence_result_schema_uri,
            semantics_uri=self.resources.graph.semantics_uri,
            payload=result_artifact_payload.model_dump(
                mode="json",
                exclude_none=True,
            ),
            parents=result_parents,
            summary=f"exact degree-sequence result: {conclusion.lower()}",
        )
        semantics = self.resources.graph.store.get(self.resources.graph.semantics_uri)
        certificate_payload = GraphDegreeSequenceReplayPayload(
            method=method,
            degree_sequence=sequence,
            conclusion=conclusion,
            graph_uri=graph_uri,
            obstruction=obstruction,
        ).model_dump(mode="json", exclude_none=True)
        certificate = CertificateEnvelope(
            certificate_type="graph.degree_sequence",
            format_version="1",
            bindings=EvidenceBindings(
                claim_digest=claim_artifact.object_digest,
                semantics_digest=semantics.manifest.object_digest,
                candidate_digest=result_artifact.object_digest,
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
                result_artifact.artifact_uri,
            ),
            summary="unverified exact degree-sequence replay certificate",
        )
        output = GraphDegreeSequenceOutput(
            degree_sequence=sequence,
            conclusion=conclusion,
            graph_uri=graph_uri,
            graph=graph_payload,
            obstruction=obstruction,
            result_uri=result_artifact.artifact_uri,
            claim_uri=claim_artifact.artifact_uri,
            certificate_uri=certificate_artifact.artifact_uri,
            backend_version=nx().__version__,
        )
        artifact_uris = [
            claim_artifact.artifact_uri,
            result_artifact.artifact_uri,
            certificate_artifact.artifact_uri,
        ]
        if graph_artifact is not None:
            artifact_uris.insert(0, graph_artifact.artifact_uri)
        return OperationProjection(
            operation_id=self.descriptor.operation_id,
            version=self.descriptor.version,
            terminal=Completed(value=output, runtime_ms=runtime_ms(started)),
            publication=PublishedOperation(
                output=output,
                artifact_uris=tuple(artifact_uris),
            ),
        )


def _degree_sequence_obstruction(
    sequence: tuple[int, ...],
) -> GraphDegreeSequenceObstruction | None:
    order = len(sequence)
    for index, degree in enumerate(sequence):
        if degree >= order:
            return GraphDegreeSequenceObstruction(
                kind="MAX_DEGREE",
                index=index,
                degree=degree,
                order=order,
            )
    degree_sum = sum(sequence)
    if degree_sum % 2:
        return GraphDegreeSequenceObstruction(
            kind="ODD_SUM",
            degree_sum=degree_sum,
        )
    ordered = sorted(sequence, reverse=True)
    for k in range(1, order + 1):
        lhs = sum(ordered[:k])
        rhs = k * (k - 1) + sum(min(degree, k) for degree in ordered[k:])
        if lhs > rhs:
            return GraphDegreeSequenceObstruction(
                kind="ERDOS_GALLAI",
                k=k,
                lhs=lhs,
                rhs=rhs,
            )
    if not nx().is_graphical(sequence, method="eg"):
        raise RuntimeError(
            "NetworkX rejected a degree sequence without a replayable obstruction"
        )
    return None
