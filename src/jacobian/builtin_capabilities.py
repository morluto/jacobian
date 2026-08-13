"""Bundled adapters for Lean capabilities."""

from __future__ import annotations

from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInvocationExample,
    CapabilityProviderRuntime,
    CapabilityRequest,
)
from jacobian.contracts.lean import (
    LeanCheckOutput,
    LeanEnvironment,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.lean_frontend.service import LeanService
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed, Failed


class LeanCheckAdapter:
    def __init__(
        self,
        lean: LeanService,
        provider_runtime: CapabilityProviderRuntime,
    ) -> None:
        self.lean = lean
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.check",
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
            input_schema={
                "type": "object",
                "properties": {
                    "statement": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "proof": {"type": "string", "minLength": 1, "maxLength": 20000},
                    "environment": {"enum": ["CORE", "MATHLIB"]},
                },
                "required": ["statement", "proof"],
                "additionalProperties": False,
            },
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
            invocation_examples=(
                CapabilityInvocationExample(
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
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> OperationProjection:
        payload = request.input
        checked = self.lean.verify(
            statement=str(payload["statement"]),
            proof=str(payload["proof"]),
            environment=LeanEnvironment(str(payload.get("environment", "CORE"))),
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
                operation_id=self.descriptor.capability_id,
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
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=Failed(
                status=execution.status,
                runtime_ms=execution.runtime_ms,
                diagnostic=CapabilityDiagnostic(
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
