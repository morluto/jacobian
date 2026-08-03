"""Operator-managed checker registry contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import model_validator

from jacobian.contracts.capabilities import (
    CapabilityId,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest
from jacobian.contracts.evidence import FormatIdentifier
from jacobian.contracts.plugins import Entrypoint
from jacobian.contracts.results import (
    Arithmetic,
    Conclusion,
    ContractModel,
    Coverage,
    Method,
)


class EvidenceKind(StrEnum):
    WITNESS = "WITNESS"
    CERTIFICATE = "CERTIFICATE"
    PRESERVATION = "PRESERVATION"
    TRANSFORMATION = "TRANSFORMATION"


class CheckerRegistration(ContractModel):
    checker_schema_version: Literal["1"] = "1"
    checker_id: CheckerUri
    name: str
    entrypoint: Entrypoint
    executable_digest: Sha256Digest
    provider_runtime: CapabilityProviderRuntime | None = None
    evidence_kind: EvidenceKind
    format_id: FormatIdentifier
    format_version: str
    claim_schema_uris: tuple[ArtifactUri, ...] = ()
    semantics_uris: tuple[ArtifactUri, ...] = ()
    candidate_schema_uris: tuple[ArtifactUri, ...] = ()
    target_schema_uris: tuple[ArtifactUri, ...] = ()
    target_semantics_uris: tuple[ArtifactUri, ...] = ()
    authorized: bool = True

    @model_validator(mode="after")
    def require_exact_external_runtime(self) -> Self:
        runtime = self.provider_runtime
        if runtime is None:
            return self
        if (
            runtime.availability is not CapabilityProviderAvailability.AVAILABLE
            or runtime.digest_kind
            not in {
                CapabilityProviderDigestKind.EXECUTABLE,
                CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD,
                CapabilityProviderDigestKind.SOURCE_TREE,
                CapabilityProviderDigestKind.COMPOSITE,
            }
            or runtime.digest is None
        ):
            raise ValueError(
                "checker provider runtime must identify an available executable, "
                "Python distribution, remeasurable source tree, or fully bound composite"
            )
        if (
            runtime.digest_kind is CapabilityProviderDigestKind.EXECUTABLE
            and not isinstance(runtime.configuration.get("executable"), str)
        ):
            raise ValueError("checker executable runtime must name its executable")
        if (
            runtime.digest_kind
            is CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD
        ):
            distribution = runtime.configuration.get("distribution")
            import_name = runtime.configuration.get("import_name")
            if not isinstance(distribution, str) or not isinstance(import_name, str):
                raise ValueError(
                    "checker Python distribution runtime must name its "
                    "distribution and import"
                )
        if runtime.digest_kind is CapabilityProviderDigestKind.SOURCE_TREE:
            entrypoint = runtime.configuration.get("entrypoint")
            if not isinstance(entrypoint, str):
                raise ValueError("checker source runtime must name its entrypoint")
            if entrypoint != self.entrypoint:
                raise ValueError(
                    "checker source runtime entrypoint must bind the checker entrypoint"
                )
        if runtime.checker_ids:
            raise ValueError(
                "checker provider runtime cannot recursively contain checker IDs"
            )
        return self


class CheckerAuditEvent(ContractModel):
    sequence: int
    checker_id: CheckerUri
    action: Literal["AUTHORIZED", "REVOKED"]
    reason: str
    recorded_at: str


class CheckerDecision(ContractModel):
    accepted: bool
    conclusion: Conclusion
    arithmetic: Arithmetic
    method: Method
    coverage: Coverage
    detail: str = ""
    relation_id: CapabilityId | None = None
    relationship_source_artifact_uris: tuple[ArtifactUri, ...] = ()
    relationship_target_artifact_uris: tuple[ArtifactUri, ...] = ()
    obligation_uri: ArtifactUri | None = None

    @model_validator(mode="after")
    def rejected_evidence_has_no_mathematical_conclusion(self) -> Self:
        if not self.accepted and self.conclusion not in {
            Conclusion.UNKNOWN,
            Conclusion.NOT_APPLICABLE,
        }:
            raise ValueError("a rejected checker input cannot decide the claim")
        if not self.accepted and (
            self.relation_id is not None
            or self.relationship_source_artifact_uris
            or self.relationship_target_artifact_uris
            or self.obligation_uri is not None
        ):
            raise ValueError("rejected evidence cannot certify relationship metadata")
        if self.relation_id is None and (
            self.relationship_source_artifact_uris
            or self.relationship_target_artifact_uris
        ):
            raise ValueError("relationship endpoints require a relation ID")
        if self.relation_id is not None and (
            not self.relationship_source_artifact_uris
            or not self.relationship_target_artifact_uris
        ):
            raise ValueError("a certified relationship requires exact endpoints")
        if len(set(self.relationship_source_artifact_uris)) != len(
            self.relationship_source_artifact_uris
        ) or len(set(self.relationship_target_artifact_uris)) != len(
            self.relationship_target_artifact_uris
        ):
            raise ValueError("certified relationship endpoints must be unique")
        if self.accepted:
            if self.conclusion not in {Conclusion.TRUE, Conclusion.FALSE}:
                raise ValueError(
                    "accepted checker evidence requires a decisive conclusion"
                )
            if self.arithmetic == Arithmetic.FLOATING_HEURISTIC:
                raise ValueError("a checker cannot accept floating heuristic evidence")
            if self.coverage in {Coverage.RESTRICTED, Coverage.SAMPLED}:
                raise ValueError(
                    "a checker cannot accept restricted or sampled evidence"
                )
            if (
                self.method == Method.DIRECT_WITNESS
                and self.coverage != Coverage.NOT_APPLICABLE
            ):
                raise ValueError("a direct witness checker cannot claim coverage")
            if (
                self.method == Method.EXHAUSTIVE_FINITE
                and self.coverage != Coverage.EXHAUSTIVE
            ):
                raise ValueError(
                    "exhaustive checker acceptance requires exhaustive coverage"
                )
        return self
