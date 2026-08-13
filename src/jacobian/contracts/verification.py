"""Immutable records produced by authorized checker execution."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from jacobian.contracts._verification_rules import (
    validate_certified_relationship_endpoints,
    validate_decisive_replayable_evidence,
)
from jacobian.contracts.checkers import CheckerManifest, EvidenceKind
from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest
from jacobian.contracts.evidence import EvidenceBindings
from jacobian.contracts.operations import OperationId
from jacobian.contracts.results import (
    Arithmetic,
    Conclusion,
    ContractModel,
    Coverage,
    Method,
)


class VerificationRecord(ContractModel):
    record_schema_version: Literal["4"] = "4"
    checker_id: CheckerUri
    implementation_digest: Sha256Digest
    checker_manifest: CheckerManifest
    evidence_kind: EvidenceKind
    evidence_uri: ArtifactUri
    bindings: EvidenceBindings
    conclusion: Conclusion
    arithmetic: Arithmetic
    method: Method
    coverage: Coverage
    request_digest: Sha256Digest
    environment_digest: Sha256Digest
    relation_id: OperationId | None = None
    relationship_source_artifact_uris: tuple[ArtifactUri, ...] = ()
    relationship_target_artifact_uris: tuple[ArtifactUri, ...] = ()
    obligation_uri: ArtifactUri | None = None

    @model_validator(mode="after")
    def require_decisive_replayable_record(self) -> Self:
        if self.implementation_digest != self.checker_manifest.implementation_digest():
            raise ValueError(
                "verification record implementation digest must match its manifest"
            )
        validate_decisive_replayable_evidence(
            self.conclusion,
            self.arithmetic,
            self.coverage,
            self.method,
        )
        validate_certified_relationship_endpoints(
            self.relation_id,
            self.relationship_source_artifact_uris,
            self.relationship_target_artifact_uris,
        )
        return self
