"""Immutable records produced by authorized checker execution."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from jacobian.contracts._verification_rules import (
    validate_certified_relationship_endpoints,
    validate_decisive_replayable_evidence,
)
from jacobian.contracts.capabilities import CapabilityId
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest
from jacobian.contracts.evidence import EvidenceBindings
from jacobian.contracts.results import (
    Arithmetic,
    Conclusion,
    ContractModel,
    Coverage,
    Method,
)


class VerificationRecord(ContractModel):
    record_schema_version: Literal["1"] = "1"
    checker_id: CheckerUri
    checker_digest: Sha256Digest
    evidence_kind: EvidenceKind
    evidence_uri: ArtifactUri
    bindings: EvidenceBindings
    conclusion: Conclusion
    arithmetic: Arithmetic
    method: Method
    coverage: Coverage
    request_digest: Sha256Digest
    environment_digest: Sha256Digest
    relation_id: CapabilityId | None = None
    relationship_source_artifact_uris: tuple[ArtifactUri, ...] = ()
    relationship_target_artifact_uris: tuple[ArtifactUri, ...] = ()
    obligation_uri: ArtifactUri | None = None

    @model_validator(mode="after")
    def require_decisive_replayable_record(self) -> Self:
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
