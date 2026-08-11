"""Typed public projections for separately authorized domain checkers."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.capability_service import CapabilityAdapter
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.common import ArtifactUri, CheckerUri
from jacobian.contracts.results import (
    ContractModel,
    Coverage,
    ExecutionStatus,
    ResultEnvelope,
    Verification,
)
from jacobian.provider_runtime import known_provider_runtime
from jacobian.schema_registry import model_schema
from jacobian.verification import VerificationService

__all__ = [
    "CertificateReplayRequest",
    "CertificateVerificationAdapter",
    "WitnessReplayRequest",
    "WitnessVerificationAdapter",
]


class CertificateReplayRequest(ContractModel):
    """One certificate whose checker identity is bound by installation."""

    certificate_uri: ArtifactUri


class WitnessReplayRequest(ContractModel):
    """One exact claim/candidate/witness tuple for a bound checker."""

    claim_uri: ArtifactUri
    candidate_uri: ArtifactUri
    witness_uri: ArtifactUri


@dataclass(frozen=True, slots=True)
class _VerificationProjection:
    capability_id: str
    title: str
    description: str
    checker_id: CheckerUri
    tags: tuple[str, ...]
    verification: VerificationService

    def descriptor(self, request_model: type[ContractModel]) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            capability_id=self.capability_id,
            version="1",
            title=self.title,
            description=self.description,
            provider="jacobian.authorized-checker",
            provider_runtime=known_provider_runtime(
                "jacobian.authorized-checker",
                features=self.tags,
                checker_ids=(self.checker_id,),
            ),
            input_schema=model_schema(request_model),
            output_schema=model_schema(ResultEnvelope),
            read_only=False,
            tags=(*self.tags, "verification"),
        )

    def result(
        self,
        request: CapabilityRequest,
        envelope: ResultEnvelope,
    ) -> CapabilityResult:
        verified = (
            envelope.execution.status is ExecutionStatus.COMPLETED
            and envelope.assurance.verification is Verification.VERIFIED
            and envelope.verification_record_uri is not None
        )
        references = set(envelope.evidence_uris)
        if envelope.assurance.scope_uri is not None:
            references.add(envelope.assurance.scope_uri)
        if envelope.verification_record_uri is not None:
            references.add(envelope.verification_record_uri)
            record = self.verification.store.get(envelope.verification_record_uri)
            references.update(record.manifest.parents)
        scope = (
            CapabilityScope(
                description=(
                    "exact artifacts replayed by the installed domain checker"
                ),
                parameters=request.input,
                artifact_uri=envelope.assurance.scope_uri,
            )
            if request.input or envelope.assurance.scope_uri is not None
            else None
        )
        complete = (
            envelope.execution.status is ExecutionStatus.COMPLETED
            and envelope.assurance.coverage is Coverage.EXHAUSTIVE
        )
        assurance_level = (
            CapabilityAssuranceLevel.VERIFIED
            if verified
            else CapabilityAssuranceLevel.HEURISTIC
        )
        return CapabilityResult(
            capability_id=self.capability_id,
            capability_version="1",
            execution=envelope.execution,
            output=envelope.model_dump(mode="json"),
            scope=scope,
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.COMPLETE
                    if complete
                    else CapabilityCompletenessStatus.PARTIAL
                ),
                basis=(
                    "the checker reported exhaustive coverage"
                    if complete
                    else "the checker made no exhaustive-coverage claim"
                ),
                assurance_level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified and complete
                    else CapabilityAssuranceLevel.COMPUTED
                ),
                verification_record_uri=(
                    envelope.verification_record_uri if verified and complete else None
                ),
            ),
            assurance=CapabilityAssurance(
                level=assurance_level,
                basis=(
                    "accepted by the installed operator-authorized checker"
                    if verified
                    else "the checker did not accept a decisive replay"
                ),
                verification_record_uri=(
                    envelope.verification_record_uri if verified else None
                ),
            ),
            artifact_uris=tuple(sorted(references)),
        )


class CertificateVerificationAdapter:
    """Replay one domain certificate with its installation-bound checker."""

    def __init__(self, projection: _VerificationProjection) -> None:
        self.projection = projection
        self._descriptor = projection.descriptor(CertificateReplayRequest)

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = CertificateReplayRequest.model_validate(request.input)
        envelope = self.projection.verification.verify_certificate(
            certificate_uri=validated.certificate_uri,
            checker_id=self.projection.checker_id,
        )
        return self.projection.result(request, envelope)


class WitnessVerificationAdapter:
    """Replay one domain witness with its installation-bound checker."""

    def __init__(self, projection: _VerificationProjection) -> None:
        self.projection = projection
        self._descriptor = projection.descriptor(WitnessReplayRequest)

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = WitnessReplayRequest.model_validate(request.input)
        envelope = self.projection.verification.verify_witness(
            claim_uri=validated.claim_uri,
            candidate_uri=validated.candidate_uri,
            witness_uri=validated.witness_uri,
            checker_id=self.projection.checker_id,
        )
        return self.projection.result(request, envelope)


def certificate_verification_adapter(
    *,
    capability_id: str,
    title: str,
    description: str,
    checker_id: CheckerUri | None,
    tags: tuple[str, ...],
    verification: VerificationService,
) -> CapabilityAdapter | None:
    if checker_id is None:
        return None
    return CertificateVerificationAdapter(
        _VerificationProjection(
            capability_id=capability_id,
            title=title,
            description=description,
            checker_id=checker_id,
            tags=tags,
            verification=verification,
        )
    )


def witness_verification_adapter(
    *,
    capability_id: str,
    title: str,
    description: str,
    checker_id: CheckerUri | None,
    tags: tuple[str, ...],
    verification: VerificationService,
) -> CapabilityAdapter | None:
    if checker_id is None:
        return None
    return WitnessVerificationAdapter(
        _VerificationProjection(
            capability_id=capability_id,
            title=title,
            description=description,
            checker_id=checker_id,
            tags=tags,
            verification=verification,
        )
    )
