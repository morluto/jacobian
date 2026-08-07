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


def _validate_rejected_checker_evidence(
    accepted: bool,
    conclusion: Conclusion,
    relation_id: CapabilityId | None,
    relationship_source_artifact_uris: tuple[ArtifactUri, ...],
    relationship_target_artifact_uris: tuple[ArtifactUri, ...],
    obligation_uri: ArtifactUri | None,
) -> None:
    if not accepted and conclusion not in {
        Conclusion.UNKNOWN,
        Conclusion.NOT_APPLICABLE,
    }:
        raise ValueError("a rejected checker input cannot decide the claim")
    if not accepted and (
        relation_id is not None
        or relationship_source_artifact_uris
        or relationship_target_artifact_uris
        or obligation_uri is not None
    ):
        raise ValueError("rejected evidence cannot certify relationship metadata")


def _validate_certified_relationship_endpoints(
    relation_id: CapabilityId | None,
    relationship_source_artifact_uris: tuple[ArtifactUri, ...],
    relationship_target_artifact_uris: tuple[ArtifactUri, ...],
) -> None:
    if relation_id is None and (
        relationship_source_artifact_uris or relationship_target_artifact_uris
    ):
        raise ValueError("relationship endpoints require a relation ID")
    if relation_id is not None and (
        not relationship_source_artifact_uris or not relationship_target_artifact_uris
    ):
        raise ValueError("a certified relationship requires exact endpoints")
    if len(set(relationship_source_artifact_uris)) != len(
        relationship_source_artifact_uris
    ) or len(set(relationship_target_artifact_uris)) != len(
        relationship_target_artifact_uris
    ):
        raise ValueError("certified relationship endpoints must be unique")


def _validate_accepted_checker_evidence(
    conclusion: Conclusion,
    arithmetic: Arithmetic,
    coverage: Coverage,
    method: Method,
) -> None:
    if conclusion not in {Conclusion.TRUE, Conclusion.FALSE}:
        raise ValueError("accepted checker evidence requires a decisive conclusion")
    if arithmetic == Arithmetic.FLOATING_HEURISTIC:
        raise ValueError("a checker cannot accept floating heuristic evidence")
    if coverage in {Coverage.RESTRICTED, Coverage.SAMPLED}:
        raise ValueError("a checker cannot accept restricted or sampled evidence")
    if method == Method.DIRECT_WITNESS and coverage != Coverage.NOT_APPLICABLE:
        raise ValueError("a direct witness checker cannot claim coverage")
    if method == Method.EXHAUSTIVE_FINITE and coverage != Coverage.EXHAUSTIVE:
        raise ValueError("exhaustive checker acceptance requires exhaustive coverage")


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
        _validate_rejected_checker_evidence(
            self.accepted,
            self.conclusion,
            self.relation_id,
            self.relationship_source_artifact_uris,
            self.relationship_target_artifact_uris,
            self.obligation_uri,
        )
        _validate_certified_relationship_endpoints(
            self.relation_id,
            self.relationship_source_artifact_uris,
            self.relationship_target_artifact_uris,
        )
        if self.accepted:
            _validate_accepted_checker_evidence(
                self.conclusion,
                self.arithmetic,
                self.coverage,
                self.method,
            )
        return self
