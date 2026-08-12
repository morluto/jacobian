"""Domain-owned graph-coloring CNF encoding evidence."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.capability_adapters import CapabilityAdapter
from jacobian.capability_errors import CapabilityInvocationError
from jacobian.checker_installation import CheckerInstaller
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityRequest,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.graph_coloring import (
    GraphColoringEncodingCandidate,
    GraphColoringEncodingClaim,
    GraphColoringEncodingOutput,
    GraphColoringEncodingReplay,
    GraphColoringEncodingRequest,
    GraphColoringEncodingScope,
)
from jacobian.domains._examples import example
from jacobian.graphs.coloring_semantics import canonical_graph, coloring_cnf
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed
from jacobian.provider_runtime import known_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.sat_smt.sat import SatArtifactService
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService
from jacobian.verification_capabilities import certificate_verification_adapter

_SEMANTICS_NAME = "jacobian.simple-undirected-graph-coloring"
_ENCODING_VERSION = "exactly-one-and-edge-separation/v1"


@dataclass(frozen=True, slots=True)
class GraphColoringInstallation:
    semantics_uri: str
    claim_schema_uri: str
    scope_schema_uri: str
    candidate_schema_uri: str
    certificate_schema_uri: str
    checker_id: str | None


def install_graph_coloring_capabilities(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    sat: SatArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[tuple[CapabilityAdapter, ...], GraphColoringInstallation]:
    """Install graph-owned coloring encodings and optional replay authorization."""

    semantics_uri = store.register_descriptor(
        kind="semantics",
        name=_SEMANTICS_NAME,
        version="1",
        definition={
            "domain": "finite simple undirected graph vertex coloring",
            "encoding": (
                "one Boolean variable per vertex/color pair; exactly one color "
                "per vertex; adjacent vertices cannot share a color"
            ),
            "encoding_version": _ENCODING_VERSION,
        },
    )
    claim_schema_uri = schemas.register_model(
        name="jacobian.graph-coloring-encoding-claim",
        version="1",
        model=GraphColoringEncodingClaim,
    )
    scope_schema_uri = schemas.register_model(
        name="jacobian.graph-coloring-encoding-scope",
        version="1",
        model=GraphColoringEncodingScope,
    )
    candidate_schema_uri = schemas.register_model(
        name="jacobian.graph-coloring-encoding-candidate",
        version="1",
        model=GraphColoringEncodingCandidate,
    )
    certificate_schema_uri = schemas.register_model(
        name="jacobian.certificate-envelope",
        version="1",
        model=CertificateEnvelope,
    )
    checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="independent graph-coloring CNF encoding checker",
                entrypoint="jacobian_checkers.graph_coloring:check_encoding",
                evidence_kind=EvidenceKind.CERTIFICATE,
                format_id="graph.coloring.encoding",
                format_version="1",
                claim_schema_uris=(claim_schema_uri,),
                semantics_uris=(semantics_uri,),
                candidate_schema_uris=(candidate_schema_uri,),
                reason=("bundled standard-library replay of graph-to-CNF semantics"),
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    installation = GraphColoringInstallation(
        semantics_uri=semantics_uri,
        claim_schema_uri=claim_schema_uri,
        scope_schema_uri=scope_schema_uri,
        candidate_schema_uri=candidate_schema_uri,
        certificate_schema_uri=certificate_schema_uri,
        checker_id=checker_id,
    )
    adapters: tuple[CapabilityAdapter, ...] = (
        GraphColoringEncodingAdapter(
            store=store,
            artifacts=artifacts,
            sat=sat,
            installation=installation,
        ),
    )
    verify = certificate_verification_adapter(
        capability_id="graph.coloring.encoding.verify",
        title="Verify a graph-coloring CNF encoding",
        description=(
            "Independently replay one exact graph-to-CNF encoding certificate."
        ),
        checker_id=checker_id,
        tags=("graph", "coloring", "cnf"),
        verification=verification,
    )
    if verify is not None:
        adapters += (verify,)
    return adapters, installation


class GraphColoringEncodingAdapter:
    """Materialize one graph-owned CNF encoding and replay certificate."""

    def __init__(
        self,
        *,
        store: ArtifactRepository,
        artifacts: ArtifactService,
        sat: SatArtifactService,
        installation: GraphColoringInstallation,
    ) -> None:
        self.store = store
        self.artifacts = artifacts
        self.sat = sat
        self.installation = installation
        self._descriptor = CapabilityDescriptor(
            capability_id="graph.coloring.encode_k_cnf",
            version="1",
            title="Encode graph k-colorability as canonical CNF",
            description=(
                "Materialize a canonical SAT instance for exactly-k vertex "
                "colorability, plus a graph-owned replay certificate."
            ),
            provider="jacobian.graph-coloring",
            provider_runtime=known_provider_runtime(
                "jacobian.graph-coloring",
                features=("k-colorability-cnf", "encoding-certificate"),
                checker_ids=(
                    (installation.checker_id,)
                    if installation.checker_id is not None
                    else ()
                ),
            ),
            input_schema=model_schema(GraphColoringEncodingRequest),
            output_schema=model_schema(GraphColoringEncodingOutput),
            tags=("graph", "coloring", "sat", "cnf", "encoding"),
            invocation_examples=(
                example(
                    "single_vertex_two_colors",
                    "Encode a one-vertex graph with two colors.",
                    {"graph": {"vertices": ["v"], "edges": []}, "colors": 2},
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> OperationProjection:
        try:
            validated = GraphColoringEncodingRequest.model_validate(request.input)
        except ValidationError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_GRAPH_COLORING_ENCODING_REQUEST",
                    stage="request_validation",
                    message="The complete graph-coloring encoding request is invalid.",
                    hint="Provide a simple graph and a color count from 1 to 32.",
                )
            ) from exc

        started = time.monotonic()
        graph = canonical_graph(validated.graph)
        variable_names, clauses = coloring_cnf(graph, validated.colors)
        cnf_artifact = self.sat.put_cnf(
            variable_names=variable_names,
            clauses=clauses,
        )
        resolved = self.sat.resolve_cnf(cnf_artifact.artifact_uri)
        claim = self.artifacts.put(
            schema_uri=self.installation.claim_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=GraphColoringEncodingClaim(
                graph=graph,
                colors=validated.colors,
            ).model_dump(mode="json"),
            summary="graph k-colorability encoding claim",
        )
        scope = self.artifacts.put(
            schema_uri=self.installation.scope_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=GraphColoringEncodingScope(
                graph=graph,
                colors=validated.colors,
                cnf_uri=cnf_artifact.artifact_uri,
                cnf_object_digest=cnf_artifact.object_digest,
                cnf=resolved.cnf,
            ).model_dump(mode="json"),
            parents=(cnf_artifact.artifact_uri,),
            summary="graph-owned canonical k-colorability CNF scope",
        )
        candidate = self.artifacts.put(
            schema_uri=self.installation.candidate_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=GraphColoringEncodingCandidate(
                cnf_uri=cnf_artifact.artifact_uri,
                scope_uri=scope.artifact_uri,
            ).model_dump(mode="json"),
            parents=(claim.artifact_uri, scope.artifact_uri),
            summary="graph k-colorability encoding candidate",
        )
        semantics = self.store.get(self.installation.semantics_uri)
        replay = GraphColoringEncodingReplay(
            claim_uri=claim.artifact_uri,
            candidate_uri=candidate.artifact_uri,
            scope_uri=scope.artifact_uri,
        ).model_dump(mode="json")
        certificate = self.artifacts.put(
            schema_uri=self.installation.certificate_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=CertificateEnvelope(
                certificate_type="graph.coloring.encoding",
                format_version="1",
                bindings=EvidenceBindings(
                    claim_digest=claim.object_digest,
                    semantics_digest=semantics.manifest.object_digest,
                    candidate_digest=candidate.object_digest,
                    scope_digest=scope.object_digest,
                ),
                payload_digest=(
                    "sha256:" + hashlib.sha256(canonicalize_json(replay)).hexdigest()
                ),
                payload=replay,
            ).model_dump(mode="json"),
            parents=(claim.artifact_uri, candidate.artifact_uri, scope.artifact_uri),
            summary="unverified graph-to-CNF replay certificate",
        )
        output = GraphColoringEncodingOutput(
            graph=graph,
            colors=validated.colors,
            cnf_uri=cnf_artifact.artifact_uri,
            scope_uri=scope.artifact_uri,
            claim_uri=claim.artifact_uri,
            candidate_uri=candidate.artifact_uri,
            certificate_uri=certificate.artifact_uri,
            variable_count=len(resolved.cnf.variables),
            clause_count=len(resolved.cnf.clauses),
        )
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=Completed(
                value=output,
                runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
            ),
            publication=PublishedOperation(
                output=output,
                artifact_uris=(
                    cnf_artifact.artifact_uri,
                    claim.artifact_uri,
                    scope.artifact_uri,
                    candidate.artifact_uri,
                    certificate.artifact_uri,
                ),
            ),
        )
