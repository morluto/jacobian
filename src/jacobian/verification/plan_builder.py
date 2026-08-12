"""Resolution and request construction for authorized verification replays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jacobian.contracts.checkers import (
    CheckerRegistration,
    EvidenceKind,
)
from jacobian.contracts.evidence import (
    CertificateEnvelope,
    EvidenceBindings,
    WitnessEnvelope,
)
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository


@dataclass(frozen=True, slots=True)
class _VerificationPlan:
    """Fully resolved replay request ready for one authorized checker run."""

    checker: CheckerRegistration
    evidence_kind: EvidenceKind
    evidence_uri: str
    bindings: EvidenceBindings
    request: dict[str, Any]
    request_artifact_uris: frozenset[str]
    parents: tuple[str, ...]
    summary: str
    claim_digest: str
    semantics_digest: str
    candidate_digest: str
    scope_uri: str | None


class VerificationPlanBuilder:
    """Resolve validated artifacts into one checker-specific replay plan."""

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        checker_registry: CheckerRegistry,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.checker_registry = checker_registry

    def build_witness(
        self,
        *,
        claim_uri: str,
        candidate_uri: str,
        witness_uri: str,
        checker_id: str,
        include_artifact_metadata: bool,
        include_semantics_artifact: bool,
    ) -> _VerificationPlan:
        """Resolve and validate one witness replay request."""

        claim = self.store.get(claim_uri)
        candidate = self.store.get(candidate_uri)
        witness_artifact = self.store.get(witness_uri)
        self._validate_artifacts((claim, candidate, witness_artifact))
        witness = WitnessEnvelope.model_validate(witness_artifact.payload)
        scope, expected_bindings, semantics_digest = self._resolve_witness_bindings(
            claim, candidate, witness_artifact, witness
        )
        self._ensure_shared_semantics(
            claim, candidate, witness_artifact, scope, label="witness"
        )
        checker = self.checker_registry.require_compatible(
            checker_id,
            evidence_kind=EvidenceKind.WITNESS,
            format_id=witness.witness_format,
            format_version=witness.format_version,
            claim_schema_uri=claim.manifest.schema_uri,
            semantics_uri=candidate.manifest.semantics_uri,
            candidate_schema_uri=candidate.manifest.schema_uri,
        )
        request = self._build_witness_request(
            claim,
            candidate,
            scope,
            witness_artifact,
            expected_bindings,
            include_artifact_metadata=include_artifact_metadata,
            include_semantics_artifact=include_semantics_artifact,
        )
        request_artifact_uris = {claim_uri, candidate_uri, witness_uri}
        if scope is not None:
            request_artifact_uris.add(scope.artifact_uri)
        return _VerificationPlan(
            checker=checker,
            evidence_kind=EvidenceKind.WITNESS,
            evidence_uri=witness_uri,
            bindings=witness.bindings,
            request=request,
            request_artifact_uris=frozenset(request_artifact_uris),
            parents=(
                claim_uri,
                candidate_uri,
                witness_uri,
                *((scope.artifact_uri,) if scope is not None else ()),
            ),
            summary="authorized witness verification",
            claim_digest=claim.manifest.object_digest,
            semantics_digest=semantics_digest,
            candidate_digest=candidate.manifest.object_digest,
            scope_uri=(scope.artifact_uri if scope is not None else None),
        )

    def build_certificate(
        self,
        *,
        certificate_uri: str,
        checker_id: str | None,
        include_artifact_metadata: bool,
        supporting_artifact_uris: tuple[str, ...],
    ) -> _VerificationPlan:
        """Resolve and validate one certificate replay request."""

        certificate_artifact = self.store.get(certificate_uri)
        self._validate_artifact(certificate_artifact)
        certificate = CertificateEnvelope.model_validate(certificate_artifact.payload)
        claim, candidate, scope = self._resolve_certificate_bindings(
            certificate_artifact, certificate
        )
        semantics_digest = self._semantics_digest(candidate)
        if certificate.bindings.semantics_digest != semantics_digest:
            raise ValueError(
                "The certificate and candidate use different semantics. Recreate "
                "the certificate from this candidate, then retry."
            )
        self._ensure_shared_semantics(
            claim, candidate, certificate_artifact, scope, label="certificate"
        )
        supporting_artifacts = self._load_supporting_artifacts(supporting_artifact_uris)
        checker = self._select_certificate_checker(
            checker_id, certificate, claim, candidate
        )
        request = self._build_certificate_request(
            claim,
            candidate,
            scope,
            certificate_artifact,
            certificate,
            supporting_artifacts,
            include_artifact_metadata=include_artifact_metadata,
        )
        request_artifact_uris = {
            claim.artifact_uri,
            candidate.artifact_uri,
            certificate_artifact.artifact_uri,
            *(artifact.artifact_uri for artifact in supporting_artifacts),
        }
        if scope is not None:
            request_artifact_uris.add(scope.artifact_uri)
        return _VerificationPlan(
            checker=checker,
            evidence_kind=EvidenceKind.CERTIFICATE,
            evidence_uri=certificate_uri,
            bindings=certificate.bindings,
            request=request,
            request_artifact_uris=frozenset(request_artifact_uris),
            parents=tuple(
                dict.fromkeys(
                    (
                        claim.artifact_uri,
                        candidate.artifact_uri,
                        certificate_uri,
                        *((scope.artifact_uri,) if scope is not None else ()),
                        *(artifact.artifact_uri for artifact in supporting_artifacts),
                    )
                )
            ),
            summary="authorized certificate verification",
            claim_digest=claim.manifest.object_digest,
            semantics_digest=semantics_digest,
            candidate_digest=candidate.manifest.object_digest,
            scope_uri=(scope.artifact_uri if scope is not None else None),
        )

    def _validate_artifact(self, artifact: StoredArtifact) -> None:
        self.schemas.validate(artifact.manifest.schema_uri, artifact.payload)

    def _validate_artifacts(self, artifacts: tuple[StoredArtifact, ...]) -> None:
        for artifact in artifacts:
            self._validate_artifact(artifact)

    def _semantics_digest(self, artifact: StoredArtifact) -> str:
        semantics = self.store.get(artifact.manifest.semantics_uri)
        return semantics.manifest.object_digest

    def _resolve_witness_bindings(
        self,
        claim: StoredArtifact,
        candidate: StoredArtifact,
        witness_artifact: StoredArtifact,
        witness: WitnessEnvelope,
    ) -> tuple[StoredArtifact | None, dict[str, Any], str]:
        if witness.bindings.encoding_digest is not None:
            raise ValueError(
                "This witness includes an unsupported encoding binding. "
                "Recreate it without encoding_digest, then retry."
            )
        scope = None
        if witness.bindings.scope_digest is not None:
            scope = self._resolve_bound_parent(
                witness_artifact,
                witness.bindings.scope_digest,
                label="scope",
            )
            self._validate_artifact(scope)
        semantics_digest = self._semantics_digest(candidate)
        expected_bindings = {
            "claim_digest": claim.manifest.object_digest,
            "semantics_digest": semantics_digest,
            "candidate_digest": candidate.manifest.object_digest,
            "scope_digest": (
                scope.manifest.object_digest if scope is not None else None
            ),
            "encoding_digest": None,
        }
        if witness.bindings.model_dump(mode="json") != expected_bindings:
            raise ValueError(
                "The witness does not match the supplied claim and candidate. "
                "Recreate the witness from those exact artifacts, then retry."
            )
        required_parents = {claim.artifact_uri, candidate.artifact_uri}
        if scope is not None:
            required_parents.add(scope.artifact_uri)
        if not required_parents.issubset(witness_artifact.manifest.parents):
            raise ValueError(
                "The witness is missing required claim or candidate lineage. "
                "Recreate it from the supplied artifacts, then retry."
            )
        return scope, expected_bindings, semantics_digest

    def _resolve_certificate_bindings(
        self,
        certificate_artifact: StoredArtifact,
        certificate: CertificateEnvelope,
    ) -> tuple[StoredArtifact, StoredArtifact, StoredArtifact | None]:
        if certificate.bindings.encoding_digest is not None:
            raise ValueError(
                "This certificate includes an unsupported encoding binding. "
                "Recreate it without encoding_digest, then retry."
            )
        claim = self._resolve_bound_parent(
            certificate_artifact,
            certificate.bindings.claim_digest,
            label="claim",
        )
        if certificate.bindings.candidate_digest is None:
            raise ValueError(
                "The certificate does not identify a candidate. Recreate it from "
                "the exact claim and candidate, then retry."
            )
        candidate = self._resolve_bound_parent(
            certificate_artifact,
            certificate.bindings.candidate_digest,
            label="candidate",
        )
        self._validate_artifacts((claim, candidate))
        scope = None
        if certificate.bindings.scope_digest is not None:
            scope = self._resolve_bound_parent(
                certificate_artifact,
                certificate.bindings.scope_digest,
                label="scope",
            )
            self._validate_artifact(scope)
        return claim, candidate, scope

    def _ensure_shared_semantics(
        self,
        claim: StoredArtifact,
        candidate: StoredArtifact,
        evidence_artifact: StoredArtifact,
        scope: StoredArtifact | None,
        *,
        label: str,
    ) -> None:
        if (
            claim.manifest.semantics_uri != candidate.manifest.semantics_uri
            or evidence_artifact.manifest.semantics_uri
            != candidate.manifest.semantics_uri
            or (
                scope is not None
                and scope.manifest.semantics_uri != candidate.manifest.semantics_uri
            )
        ):
            raise ValueError(
                f"The claim, candidate, {label}, and scope use different "
                "semantics. Use artifacts from one reference contract, then retry."
            )

    def _load_supporting_artifacts(
        self, supporting_artifact_uris: tuple[str, ...]
    ) -> tuple[StoredArtifact, ...]:
        artifacts = tuple(
            self.store.get(uri) for uri in dict.fromkeys(supporting_artifact_uris)
        )
        self._validate_artifacts(artifacts)
        return artifacts

    def _select_certificate_checker(
        self,
        checker_id: str | None,
        certificate: CertificateEnvelope,
        claim: StoredArtifact,
        candidate: StoredArtifact,
    ) -> CheckerRegistration:
        compatibility: dict[str, Any] = {
            "evidence_kind": EvidenceKind.CERTIFICATE,
            "format_id": certificate.certificate_type,
            "format_version": certificate.format_version,
            "claim_schema_uri": claim.manifest.schema_uri,
            "semantics_uri": candidate.manifest.semantics_uri,
            "candidate_schema_uri": candidate.manifest.schema_uri,
        }
        if checker_id is None:
            return self.checker_registry.select_compatible(**compatibility)
        return self.checker_registry.require_compatible(checker_id, **compatibility)

    def _build_witness_request(
        self,
        claim: StoredArtifact,
        candidate: StoredArtifact,
        scope: StoredArtifact | None,
        witness_artifact: StoredArtifact,
        expected_bindings: dict[str, Any],
        *,
        include_artifact_metadata: bool,
        include_semantics_artifact: bool,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "request_version": "1",
            "claim": _checker_artifact(
                claim,
                include_storage_metadata=include_artifact_metadata,
            ),
            "candidate": _checker_artifact(
                candidate,
                include_storage_metadata=include_artifact_metadata,
            ),
            "scope": (
                _checker_artifact(
                    scope,
                    include_storage_metadata=include_artifact_metadata,
                )
                if scope is not None
                else None
            ),
            "witness": _checker_artifact(
                witness_artifact,
                include_storage_metadata=include_artifact_metadata,
            ),
            "expected_bindings": expected_bindings,
        }
        if include_semantics_artifact:
            semantics_artifact = self.store.get(candidate.manifest.semantics_uri)
            request["semantics"] = _checker_artifact(
                semantics_artifact,
                include_storage_metadata=include_artifact_metadata,
            )
        return request

    def _build_certificate_request(
        self,
        claim: StoredArtifact,
        candidate: StoredArtifact,
        scope: StoredArtifact | None,
        certificate_artifact: StoredArtifact,
        certificate: CertificateEnvelope,
        supporting_artifacts: tuple[StoredArtifact, ...],
        *,
        include_artifact_metadata: bool,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "request_version": "1",
            "claim": _checker_artifact(
                claim,
                include_storage_metadata=include_artifact_metadata,
            ),
            "candidate": _checker_artifact(
                candidate,
                include_storage_metadata=include_artifact_metadata,
            ),
            "scope": (
                _checker_artifact(
                    scope,
                    include_storage_metadata=include_artifact_metadata,
                )
                if scope is not None
                else None
            ),
            "certificate": {
                **_checker_artifact(
                    certificate_artifact,
                    include_storage_metadata=include_artifact_metadata,
                ),
                "payload": certificate.model_dump(mode="json"),
            },
            "expected_bindings": certificate.bindings.model_dump(mode="json"),
        }
        if supporting_artifacts:
            request["supporting_artifacts"] = [
                _checker_artifact(
                    artifact,
                    include_storage_metadata=include_artifact_metadata,
                )
                for artifact in supporting_artifacts
            ]
        return request

    def _resolve_bound_parent(
        self,
        evidence_artifact: StoredArtifact,
        object_digest: str,
        *,
        label: str,
    ) -> StoredArtifact:
        parent_set = set(evidence_artifact.manifest.parents)
        matches = [
            uri
            for uri in self.store.find_by_object_digest(object_digest)
            if uri in parent_set
        ]
        if not matches:
            raise ValueError(
                f"The certificate is missing its bound {label} artifact. Recreate "
                "the certificate from the exact verification inputs, then retry."
            )
        return self.store.get(sorted(matches)[0])


def _checker_artifact(
    artifact: StoredArtifact | None,
    *,
    include_storage_metadata: bool = False,
) -> dict[str, Any]:
    if artifact is None:
        raise ValueError(
            "Verification evidence is incomplete. Recreate it from the exact "
            "claim and candidate, then retry."
        )
    result: dict[str, Any] = {
        "artifact_uri": artifact.artifact_uri,
        "object_digest": artifact.manifest.object_digest,
        "schema_uri": artifact.manifest.schema_uri,
        "semantics_uri": artifact.manifest.semantics_uri,
        "payload": artifact.payload,
    }
    if include_storage_metadata:
        result["payload_digest"] = artifact.manifest.payload_digest
        result["parents"] = list(artifact.manifest.parents)
    return result
