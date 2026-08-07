"""Checker-backed verification of one explicit graph-isomorphism mapping."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.capability_service import CapabilityInvocationError
from jacobian.checker_installation import CheckerInstaller
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityMode,
    CapabilityRelationship,
    CapabilityRelationshipStatus,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.graph_isomorphism import (
    GraphIsomorphismClaim,
    GraphIsomorphismReplay,
    GraphIsomorphismVerifyOutput,
    GraphIsomorphismVerifyRequest,
    GraphPair,
    GraphVertexMapping,
    SimpleUndirectedGraph,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.graphs.installation import GraphInstallation
from jacobian.provider_runtime import known_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService


@dataclass(frozen=True, slots=True)
class GraphIsomorphismInstallation:
    semantics_uri: str
    source_graph_semantics_uri: str
    source_graph_schema_uri: str
    pair_schema_uri: str
    mapping_schema_uri: str
    claim_schema_uri: str
    certificate_schema_uri: str
    checker_id: str | None


@dataclass(frozen=True, slots=True)
class _SourceGraph:
    object_digest: str
    graph: SimpleUndirectedGraph


def install_graph_isomorphism(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    graph: GraphInstallation,
    *,
    authorize_checker: bool,
) -> tuple[GraphIsomorphismAdapter | None, GraphIsomorphismInstallation]:
    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.simple-undirected-graph-isomorphism",
        version="1",
        definition={
            "description": (
                "isomorphism of two finite simple undirected graphs through one "
                "explicit vertex bijection"
            ),
            "maximum_order": 256,
        },
    )
    pair_schema_uri = schemas.register(
        name="jacobian.simple-graph-pair",
        version="1",
        schema=model_schema(GraphPair),
    )
    mapping_schema_uri = schemas.register(
        name="jacobian.graph-vertex-mapping",
        version="1",
        schema=model_schema(GraphVertexMapping),
    )
    claim_schema_uri = schemas.register(
        name="jacobian.graph-isomorphism-claim",
        version="1",
        schema=model_schema(GraphIsomorphismClaim),
    )
    certificate_schema_uri = schemas.register(
        name="jacobian.certificate-envelope",
        version="1",
        schema=model_schema(CertificateEnvelope),
    )
    checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="exact finite simple-graph isomorphism checker",
                entrypoint="jacobian_checkers.graph_isomorphism:check_isomorphism",
                evidence_kind=EvidenceKind.CERTIFICATE,
                format_id="graph.isomorphism_replay",
                format_version="1",
                claim_schema_uris=(claim_schema_uri,),
                semantics_uris=(semantics_uri,),
                candidate_schema_uris=(mapping_schema_uri,),
                reason="bundled independent adjacency-preservation checker",
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    installation = GraphIsomorphismInstallation(
        semantics_uri=semantics_uri,
        source_graph_semantics_uri=graph.semantics_uri,
        source_graph_schema_uri=graph.graph_schema_uri,
        pair_schema_uri=pair_schema_uri,
        mapping_schema_uri=mapping_schema_uri,
        claim_schema_uri=claim_schema_uri,
        certificate_schema_uri=certificate_schema_uri,
        checker_id=checker_id,
    )
    if checker_id is None:
        return None, installation
    return (
        GraphIsomorphismAdapter(
            store=store,
            artifacts=artifacts,
            verification=verification,
            installation=installation,
        ),
        installation,
    )


class GraphIsomorphismAdapter:
    def __init__(
        self,
        *,
        store: ArtifactRepository,
        artifacts: ArtifactService,
        verification: VerificationService,
        installation: GraphIsomorphismInstallation,
    ) -> None:
        self.store = store
        self.artifacts = artifacts
        self.verification = verification
        self.installation = installation
        if installation.checker_id is None:
            raise RuntimeError(
                "graph isomorphism verify adapter requires an authorized checker"
            )
        self._descriptor = CapabilityDescriptor(
            capability_id="graph.isomorphism.verify",
            version="1",
            title="Verify one graph-isomorphism mapping",
            description=(
                "Independently check that one explicit vertex bijection preserves "
                "all adjacency and nonadjacency."
            ),
            provider="jacobian.graph-isomorphism-checker",
            provider_runtime=known_provider_runtime(
                "jacobian.graph-isomorphism-checker",
                features=("graph", "isomorphism", "direct-witness"),
                checker_ids=(installation.checker_id,),
            ),
            modes=(CapabilityMode.VERIFY,),
            input_schema=model_schema(GraphIsomorphismVerifyRequest),
            output_schema=model_schema(GraphIsomorphismVerifyOutput),
            tags=("graph", "isomorphism", "verification"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = GraphIsomorphismVerifyRequest.model_validate(request.input)
        checker_id = self.installation.checker_id
        if checker_id is None:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="GRAPH_ISOMORPHISM_CHECKER_UNAVAILABLE",
                    stage="isomorphism_verification",
                    message=(
                        "The independent graph isomorphism checker is not installed "
                        "in this runtime."
                    ),
                )
            )
        left = self._load_source_graph(
            validated.left_graph_uri,
            path="left_graph_uri",
        )
        right = self._load_source_graph(
            validated.right_graph_uri,
            path="right_graph_uri",
        )
        source_graph_uris = tuple(
            dict.fromkeys((validated.left_graph_uri, validated.right_graph_uri))
        )
        pair = self.artifacts.put(
            schema_uri=self.installation.pair_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=GraphPair(
                left_graph_uri=validated.left_graph_uri,
                right_graph_uri=validated.right_graph_uri,
                left_graph_digest=left.object_digest,
                right_graph_digest=right.object_digest,
                graph_schema_uri=self.installation.source_graph_schema_uri,
                graph_semantics_uri=self.installation.source_graph_semantics_uri,
                left=left.graph,
                right=right.graph,
            ).model_dump(mode="json"),
            parents=source_graph_uris,
            summary="finite simple-graph pair",
        )
        mapping = self.artifacts.put(
            schema_uri=self.installation.mapping_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=GraphVertexMapping(mapping=validated.mapping).model_dump(
                mode="json"
            ),
            summary="proposed graph vertex mapping",
        )
        claim = self.artifacts.put(
            schema_uri=self.installation.claim_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=GraphIsomorphismClaim(
                graph_pair_uri=pair.artifact_uri,
                mapping_uri=mapping.artifact_uri,
            ).model_dump(mode="json"),
            parents=(pair.artifact_uri, mapping.artifact_uri),
            summary="explicit graph-isomorphism mapping claim",
        )
        semantics = self.store.get(self.installation.semantics_uri)
        replay = GraphIsomorphismReplay(
            graph_pair_uri=pair.artifact_uri,
            mapping_uri=mapping.artifact_uri,
            left_graph_uri=validated.left_graph_uri,
            right_graph_uri=validated.right_graph_uri,
            left_graph_digest=left.object_digest,
            right_graph_digest=right.object_digest,
            graph_schema_uri=self.installation.source_graph_schema_uri,
            graph_semantics_uri=self.installation.source_graph_semantics_uri,
        ).model_dump(mode="json")
        certificate = CertificateEnvelope(
            certificate_type="graph.isomorphism_replay",
            format_version="1",
            bindings=EvidenceBindings(
                claim_digest=claim.object_digest,
                semantics_digest=semantics.manifest.object_digest,
                candidate_digest=mapping.object_digest,
                scope_digest=pair.object_digest,
            ),
            payload_digest=(
                "sha256:" + hashlib.sha256(canonicalize_json(replay)).hexdigest()
            ),
            payload=replay,
        )
        evidence = self.artifacts.put(
            schema_uri=self.installation.certificate_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=(claim.artifact_uri, mapping.artifact_uri, pair.artifact_uri),
            summary="graph-isomorphism adjacency replay certificate",
        )
        checked = self.verification.verify_certificate(
            certificate_uri=evidence.artifact_uri,
            checker_id=checker_id,
            supporting_artifact_uris=source_graph_uris,
        )
        verified = (
            checked.execution.status is ExecutionStatus.COMPLETED
            and checked.verification_record_uri is not None
        )
        conclusion = cast(
            Literal["TRUE", "FALSE", "UNKNOWN"],
            checked.conclusion.value,
        )
        is_isomorphism = {
            "TRUE": True,
            "FALSE": False,
            "UNKNOWN": None,
        }[conclusion]
        output = GraphIsomorphismVerifyOutput(
            is_isomorphism=is_isomorphism,
            conclusion=conclusion,
            left_graph_uri=validated.left_graph_uri,
            right_graph_uri=validated.right_graph_uri,
            graph_pair_uri=pair.artifact_uri,
            mapping_uri=mapping.artifact_uri,
            claim_uri=claim.artifact_uri,
            certificate_uri=evidence.artifact_uri,
            verification_record_uri=checked.verification_record_uri,
            checker_id=checker_id,
            coverage="EXHAUSTIVE" if verified else "UNKNOWN",
        )
        record_uri = checked.verification_record_uri
        artifact_uris = [
            *source_graph_uris,
            pair.artifact_uri,
            mapping.artifact_uri,
            claim.artifact_uri,
            evidence.artifact_uri,
        ]
        if record_uri is not None:
            artifact_uris.append(record_uri)
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=checked.execution,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="every vertex pair in two explicit finite graphs",
                parameters={
                    "left_graph_uri": validated.left_graph_uri,
                    "right_graph_uri": validated.right_graph_uri,
                    "left_order": len(left.graph.vertices),
                    "right_order": len(right.graph.vertices),
                },
                artifact_uri=pair.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.COMPLETE
                    if verified
                    else CapabilityCompletenessStatus.UNKNOWN
                ),
                basis=(
                    "the checker compared the complete finite adjacency relation"
                    if verified
                    else "the checker did not accept the replay inputs"
                ),
                assurance_level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
                verification_record_uri=record_uri,
            ),
            relationships=(
                CapabilityRelationship(
                    relation_id="graph.relation.pair-scope",
                    source_artifact_uris=source_graph_uris,
                    target_artifact_uris=(pair.artifact_uri,),
                ),
                *(
                    (
                        CapabilityRelationship(
                            relation_id="graph.relation.isomorphic-via",
                            source_artifact_uris=source_graph_uris,
                            target_artifact_uris=(mapping.artifact_uri,),
                            status=CapabilityRelationshipStatus.VERIFIED,
                            verification_record_uri=record_uri,
                        ),
                    )
                    if conclusion == "TRUE" and record_uri is not None
                    else ()
                ),
            ),
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
                basis=(
                    "accepted by the authorized independent adjacency checker"
                    if verified
                    else "the independent checker did not accept the mapping"
                ),
                verification_record_uri=record_uri,
            ),
            artifact_uris=tuple(artifact_uris),
        )

    def _load_source_graph(
        self,
        graph_uri: str,
        *,
        path: Literal["left_graph_uri", "right_graph_uri"],
    ) -> _SourceGraph:
        try:
            artifact = self.store.get(graph_uri)
        except StorageError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="GRAPH_ARTIFACT_NOT_FOUND",
                    stage="graph_resolution",
                    message=f"The graph artifact at {path} is unavailable.",
                    path=path,
                    hint=(
                        "Use a graph URI returned by graph.search.atlas or another "
                        "installed graph capability."
                    ),
                )
            ) from exc
        if (
            artifact.manifest.schema_uri != self.installation.source_graph_schema_uri
            or artifact.manifest.semantics_uri
            != self.installation.source_graph_semantics_uri
        ):
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INCOMPATIBLE_GRAPH_ARTIFACT",
                    stage="graph_validation",
                    message=(
                        f"The artifact at {path} is not a compatible simple "
                        "undirected graph."
                    ),
                    path=path,
                    schema_uri=self.installation.source_graph_schema_uri,
                    hint=(
                        "Use a graph URI returned by graph.search.atlas or another "
                        "installed graph capability."
                    ),
                )
            )
        try:
            graph = SimpleUndirectedGraph.model_validate(artifact.payload)
        except ValidationError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INCOMPATIBLE_GRAPH_ARTIFACT",
                    stage="graph_validation",
                    message=f"The graph artifact at {path} has a malformed payload.",
                    path=path,
                    schema_uri=self.installation.source_graph_schema_uri,
                    hint="Recreate the graph through its owning capability.",
                )
            ) from exc
        return _SourceGraph(
            object_digest=artifact.manifest.object_digest,
            graph=graph,
        )
