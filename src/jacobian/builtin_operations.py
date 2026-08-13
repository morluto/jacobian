"""Bundled adapters for Lean operations."""

from __future__ import annotations

from jacobian.contracts.lean import (
    LeanCheckOutput,
    LeanCheckRequest,
)
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiagnostic,
    OperationExample,
    OperationRequest,
    ProviderObservation,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.lean_frontend.service import LeanService
from jacobian.operation_adapters import parse_operation_input
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed, Failed


class LeanCheckAdapter:
    def __init__(
        self,
        lean: LeanService,
        provider_runtime: ProviderObservation,
    ) -> None:
        self.lean = lean
        self._descriptor = OperationDescriptor(
            operation_id="lean.check",
            version="2",
            title="Independently check an exact Lean proof",
            description=(
                "Compile and replay one proposition with the pinned CORE or MATHLIB "
                "kernel profile. Statements are single Lean expressions, including "
                "finite-witness let expressions; declarations and trust escapes are "
                "forbidden. Rejections include stable proof-repair diagnostics, "
                "payload-relative source spans, and the bounded backend message."
            ),
            provider="jacobian.lean4",
            provider_runtime=provider_runtime,
            input_schema=LeanCheckRequest.model_json_schema(),
            output_schema=LeanCheckOutput.model_json_schema(),
            tags=(
                "lean",
                "proof",
                "checker",
                "verification",
                "core",
                "mathlib",
                "finite-witness",
                "proof-repair",
                "diagnostics",
                "type-mismatch",
                "source-span",
            ),
            examples=(
                OperationExample(
                    name="finite-witness-let",
                    description=(
                        "Check a finite witness encoded as one let expression without "
                        "adding declarations."
                    ),
                    input={
                        "environment": "CORE",
                        "statement": "let n : Nat := 2; n + n = 4",
                        "proof": "rfl",
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> LeanCheckRequest:
        return parse_operation_input(LeanCheckRequest, request.input)

    def invoke(self, payload: LeanCheckRequest) -> OperationProjection:
        checked = self.lean.verify(
            statement=payload.statement,
            proof=payload.proof,
            environment=payload.environment,
        )
        verified = checked.result.verification_record_uri is not None
        evidence = (checked.certificate_uri,)
        scope_uri = checked.result.scope_uri
        output = LeanCheckOutput(
            conclusion=checked.result.conclusion,
            execution=checked.result.execution,
            input=checked.result.input,
            diagnostics=checked.diagnostics,
            claim_uri=checked.claim_uri,
            candidate_uri=checked.candidate_uri,
            certificate_uri=checked.certificate_uri,
            verification_record_uri=checked.result.verification_record_uri,
            cache_hit=checked.cache_hit,
        )
        record_uri = checked.result.verification_record_uri if verified else None
        publication = PublishedOperation(
            output=output,
            artifact_uris=(
                checked.claim_uri,
                checked.candidate_uri,
                *evidence,
                *((scope_uri,) if scope_uri is not None else ()),
                *((record_uri,) if record_uri is not None else ()),
            ),
        )
        execution = checked.result.execution
        if execution.status is ExecutionStatus.COMPLETED:
            return OperationProjection(
                operation_id=self.descriptor.operation_id,
                version=self.descriptor.version,
                terminal=Completed(
                    value=output,
                    runtime_ms=execution.runtime_ms,
                    detail=execution.detail,
                ),
                publication=publication,
                verification_record_uri=record_uri,
            )
        return OperationProjection(
            operation_id=self.descriptor.operation_id,
            version=self.descriptor.version,
            terminal=Failed(
                status=execution.status,
                runtime_ms=execution.runtime_ms,
                diagnostic=OperationDiagnostic(
                    code="LEAN_CHECK_NONCONCLUSIVE",
                    stage="verification",
                    message=(
                        execution.detail
                        or "The independent Lean checker reached no conclusion."
                    ),
                ),
            ),
            publication=PublishedOperation(artifact_uris=publication.artifact_uris),
        )
