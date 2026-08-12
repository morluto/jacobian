"""Typed public projections for separately authorized domain checkers."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from jacobian.capability_errors import (
    CapabilityInvocationError,
    enriched_invalid_request,
)
from jacobian.capability_service import CapabilityAdapter
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.common import ArtifactUri, CheckerUri
from jacobian.contracts.results import (
    ContractModel,
    ExecutionStatus,
    VerificationResult,
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


_INVALID_REPLAY_REQUEST = CapabilityDiagnostic(
    code="INVALID_REQUEST",
    stage="capability_input_validation",
    message="The checker replay request is invalid.",
    hint="Inspect the checker operation and retry with its required artifact inputs.",
)


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
            output_schema=model_schema(VerificationResult),
            read_only=False,
            tags=(*self.tags, "verification"),
        )

    def result(
        self,
        result: VerificationResult,
    ) -> CapabilityResult:
        verified = (
            result.execution.status is ExecutionStatus.COMPLETED
            and result.verification_record_uri is not None
        )
        references = set(result.evidence_uris)
        if result.scope_uri is not None:
            references.add(result.scope_uri)
        if result.verification_record_uri is not None:
            references.add(result.verification_record_uri)
            record = self.verification.store.get(result.verification_record_uri)
            references.update(record.manifest.parents)
        return CapabilityResult(
            capability_id=self.capability_id,
            capability_version="1",
            execution=result.execution,
            output=result.model_dump(mode="json"),
            verification_record_uri=(
                result.verification_record_uri if verified else None
            ),
            artifact_uris=tuple(sorted(references)),
        )


class CertificateVerificationAdapter:
    """Replay one domain certificate with its installation-bound checker."""

    typed_input = True

    def __init__(self, projection: _VerificationProjection) -> None:
        self.projection = projection
        self._descriptor = projection.descriptor(CertificateReplayRequest)

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = CertificateReplayRequest.model_validate(request.input)
        except ValidationError as exc:
            raise CapabilityInvocationError(
                enriched_invalid_request(_INVALID_REPLAY_REQUEST, exc)
            ) from exc
        result = self.projection.verification.verify_certificate(
            certificate_uri=validated.certificate_uri,
            checker_id=self.projection.checker_id,
        )
        return self.projection.result(result)


class WitnessVerificationAdapter:
    """Replay one domain witness with its installation-bound checker."""

    typed_input = True

    def __init__(self, projection: _VerificationProjection) -> None:
        self.projection = projection
        self._descriptor = projection.descriptor(WitnessReplayRequest)

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = WitnessReplayRequest.model_validate(request.input)
        except ValidationError as exc:
            raise CapabilityInvocationError(
                enriched_invalid_request(_INVALID_REPLAY_REQUEST, exc)
            ) from exc
        result = self.projection.verification.verify_witness(
            claim_uri=validated.claim_uri,
            candidate_uri=validated.candidate_uri,
            witness_uri=validated.witness_uri,
            checker_id=self.projection.checker_id,
        )
        return self.projection.result(result)


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
