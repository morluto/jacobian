"""Typed public projections for separately authorized domain checkers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from jacobian.contracts.common import ArtifactUri, CheckerUri
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiagnostic,
    OperationRequest,
)
from jacobian.contracts.results import (
    ContractModel,
    ExecutionStatus,
    VerificationResult,
)
from jacobian.operation_adapters import OperationAdapter, parse_operation_input
from jacobian.operation_errors import (
    OperationInvocationError,
    enriched_invalid_request,
)
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed, Failed
from jacobian.provider_runtime import known_provider_runtime
from jacobian.schema_registry import model_schema
from jacobian.verification.service import VerificationService

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


_INVALID_REPLAY_REQUEST = OperationDiagnostic(
    code="INVALID_REQUEST",
    stage="operation_input_validation",
    message="The checker replay request is invalid.",
    hint="Inspect the checker operation and retry with its required artifact inputs.",
)


@dataclass(frozen=True, slots=True)
class _VerificationProjection:
    operation_id: str
    title: str
    description: str
    checker_id: CheckerUri
    tags: tuple[str, ...]
    verification: VerificationService

    def descriptor(self, request_model: type[ContractModel]) -> OperationDescriptor:
        return OperationDescriptor(
            operation_id=self.operation_id,
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
    ) -> OperationProjection:
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
        record_uri = result.verification_record_uri if verified else None
        terminal = (
            Completed(
                value=result,
                runtime_ms=result.execution.runtime_ms,
                detail=result.execution.detail,
            )
            if result.execution.status is ExecutionStatus.COMPLETED
            else Failed(
                status=result.execution.status,
                runtime_ms=result.execution.runtime_ms,
                diagnostic=OperationDiagnostic(
                    code="CHECKER_REPLAY_EXECUTION_FAILED",
                    stage="checker_replay",
                    message=(
                        result.execution.detail
                        or (result.input.errors[0] if result.input.errors else None)
                        or "The authorized checker did not complete the replay."
                    ),
                ),
            )
        )
        return OperationProjection(
            operation_id=self.operation_id,
            version="1",
            terminal=terminal,
            publication=PublishedOperation(
                output=result if isinstance(terminal, Completed) else None,
                artifact_uris=tuple(sorted(references)),
            ),
            verification_record_uri=record_uri,
        )


class CertificateVerificationAdapter:
    """Replay one domain certificate with its installation-bound checker."""

    def __init__(self, projection: _VerificationProjection) -> None:
        self.projection = projection
        self._descriptor = projection.descriptor(CertificateReplayRequest)

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> CertificateReplayRequest:
        try:
            return parse_operation_input(CertificateReplayRequest, request.input)
        except ValidationError as exc:
            raise OperationInvocationError(
                enriched_invalid_request(_INVALID_REPLAY_REQUEST, exc)
            ) from exc

    def invoke(self, validated: CertificateReplayRequest) -> OperationProjection:
        result = self.projection.verification.verify_certificate(
            certificate_uri=validated.certificate_uri,
            checker_id=self.projection.checker_id,
        )
        return self.projection.result(result)


class WitnessVerificationAdapter:
    """Replay one domain witness with its installation-bound checker."""

    def __init__(self, projection: _VerificationProjection) -> None:
        self.projection = projection
        self._descriptor = projection.descriptor(WitnessReplayRequest)

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> WitnessReplayRequest:
        try:
            return parse_operation_input(WitnessReplayRequest, request.input)
        except ValidationError as exc:
            raise OperationInvocationError(
                enriched_invalid_request(_INVALID_REPLAY_REQUEST, exc)
            ) from exc

    def invoke(self, validated: WitnessReplayRequest) -> OperationProjection:
        result = self.projection.verification.verify_witness(
            claim_uri=validated.claim_uri,
            candidate_uri=validated.candidate_uri,
            witness_uri=validated.witness_uri,
            checker_id=self.projection.checker_id,
        )
        return self.projection.result(result)


def certificate_verification_adapter(
    *,
    operation_id: str,
    title: str,
    description: str,
    checker_id: CheckerUri | None,
    tags: tuple[str, ...],
    verification: VerificationService,
) -> OperationAdapter[Any] | None:
    if checker_id is None:
        return None
    return CertificateVerificationAdapter(
        _VerificationProjection(
            operation_id=operation_id,
            title=title,
            description=description,
            checker_id=checker_id,
            tags=tags,
            verification=verification,
        )
    )


def witness_verification_adapter(
    *,
    operation_id: str,
    title: str,
    description: str,
    checker_id: CheckerUri | None,
    tags: tuple[str, ...],
    verification: VerificationService,
) -> OperationAdapter[Any] | None:
    if checker_id is None:
        return None
    return WitnessVerificationAdapter(
        _VerificationProjection(
            operation_id=operation_id,
            title=title,
            description=description,
            checker_id=checker_id,
            tags=tags,
            verification=verification,
        )
    )
