"""Bundled adapters for Lean capabilities."""

from __future__ import annotations

from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityInvocationExample,
    CapabilityProviderRuntime,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.lean import (
    LeanCheckOutput,
    LeanEnvironment,
)
from jacobian.lean_frontend.service import LeanService


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

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
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
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=checked.result.execution,
            output=output.model_dump(mode="json"),
            verification_record_uri=(
                checked.result.verification_record_uri if verified else None
            ),
            artifact_uris=(
                checked.claim_uri,
                checked.candidate_uri,
                *evidence,
                *((scope_uri,) if scope_uri is not None else ()),
                *(
                    (checked.result.verification_record_uri,)
                    if checked.result.verification_record_uri is not None
                    else ()
                ),
            ),
        )
