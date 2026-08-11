"""Checker-backed validation of exact Lean proof edits."""

from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityProviderRuntime,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.lean_proof_edit import (
    LeanProofEditArtifact,
    LeanProofEditOutput,
    LeanProofEditRequest,
)
from jacobian.contracts.results import (
    Conclusion,
    ExecutionStatus,
    ResultEnvelope,
    Verification,
)
from jacobian.lean_frontend.service import LeanService
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository

_FORBIDDEN_PROOF_HOLE = re.compile(r"\b(?:admit|sorry)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class LeanProofEditInstallation:
    semantics_uri: str
    artifact_schema_uri: str


def install_lean_proof_edit_capability(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    lean: LeanService,
    provider_runtime: CapabilityProviderRuntime,
) -> tuple[LeanProofEditAdapter, LeanProofEditInstallation]:
    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.lean4-proof-edit-validation",
        version="2",
        definition={
            "description": (
                "an exact Lean proof edit checked against the unchanged statement "
                "by the operator-authorized lean.check checker"
            ),
            "proof_holes": "sorry and admit are rejected before checker invocation",
            "verification": "delegated exclusively to lean.check",
        },
    )
    schema_uri = schemas.register(
        name="jacobian.lean4-proof-edit",
        version="2",
        schema=LeanProofEditArtifact.model_json_schema(),
    )
    installation = LeanProofEditInstallation(
        semantics_uri=semantics_uri,
        artifact_schema_uri=schema_uri,
    )
    return (
        LeanProofEditAdapter(
            lean,
            artifacts,
            provider_runtime,
            installation,
        ),
        installation,
    )


class LeanProofEditAdapter:
    def __init__(
        self,
        lean: LeanService,
        artifacts: ArtifactService,
        provider_runtime: CapabilityProviderRuntime,
        installation: LeanProofEditInstallation,
    ) -> None:
        self.lean = lean
        self.artifacts = artifacts
        self.installation = installation
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.proof_edit.validate",
            version="2",
            title="Validate an exact Lean proof edit",
            description=(
                "Bind an original and edited proof to one unchanged statement, then "
                "submit the exact edited source through the operator-authorized "
                "lean.check checker."
            ),
            provider="jacobian.lean4",
            provider_runtime=provider_runtime,
            input_schema=LeanProofEditRequest.model_json_schema(),
            output_schema=LeanProofEditOutput.model_json_schema(),
            tags=("lean", "proof-edit", "validation", "checker"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = LeanProofEditRequest.model_validate(request.input)
            if _FORBIDDEN_PROOF_HOLE.search(
                validated.original_proof
            ) or _FORBIDDEN_PROOF_HOLE.search(validated.edited_proof):
                raise ValueError("proof edit contains a proof hole")
        except (ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_LEAN_PROOF_EDIT_REQUEST",
                    stage="request_validation",
                    message="The exact Lean proof-edit request is invalid.",
                    hint=(
                        "Keep the statement unchanged, provide distinct original and "
                        "edited proof bodies, and remove sorry or admit."
                    ),
                )
            ) from exc
        started = time.monotonic()
        baseline = self.lean.verify(
            environment=validated.environment,
            statement=validated.statement,
            proof=validated.original_proof,
        )
        checked = self.lean.verify(
            environment=validated.environment,
            statement=validated.statement,
            proof=validated.edited_proof,
        )
        baseline_verified = _is_verified(baseline.result)
        verified = baseline_verified and _is_verified(checked.result)
        diff = "".join(
            difflib.unified_diff(
                validated.original_proof.splitlines(keepends=True),
                validated.edited_proof.splitlines(keepends=True),
                fromfile="original_proof",
                tofile="edited_proof",
            )
        )
        payload = LeanProofEditArtifact(
            environment=validated.environment,
            statement=validated.statement,
            original_proof=validated.original_proof,
            edited_proof=validated.edited_proof,
            unified_diff=diff,
            baseline_checker_execution_status=baseline.result.execution.status,
            baseline_accepted=baseline_verified,
            baseline_candidate_uri=baseline.candidate_uri,
            baseline_certificate_uri=baseline.certificate_uri,
            baseline_verification_record_uri=(baseline.result.verification_record_uri),
            checker_execution_status=checked.result.execution.status,
            accepted=verified,
            claim_uri=checked.claim_uri,
            candidate_uri=checked.candidate_uri,
            certificate_uri=checked.certificate_uri,
            verification_record_uri=checked.result.verification_record_uri,
        )
        artifact = self.artifacts.put(
            schema_uri=self.installation.artifact_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=payload.model_dump(mode="json"),
            parents=(
                checked.claim_uri,
                baseline.candidate_uri,
                baseline.certificate_uri,
                checked.candidate_uri,
                checked.certificate_uri,
            ),
            summary=f"Lean proof edit validation (accepted={verified})",
        )
        output = LeanProofEditOutput(
            **payload.model_dump(mode="python"),
            proof_edit_uri=artifact.artifact_uri,
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=checked.result.execution.model_copy(
                update={"runtime_ms": int((time.monotonic() - started) * 1000)}
            ),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="one exact edited Lean proof for one unchanged statement",
                parameters={
                    "environment": validated.environment.value,
                    "statement": validated.statement,
                },
                artifact_uri=artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.NOT_APPLICABLE,
                basis=(
                    "direct proof replay makes no search-completeness claim"
                    if checked.result.execution.status is ExecutionStatus.COMPLETED
                    else "the Lean checker did not complete; completeness is not applicable"
                ),
                assurance_level=(
                    CapabilityAssuranceLevel.COMPUTED
                    if checked.result.execution.status is ExecutionStatus.COMPLETED
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
            ),
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
                basis=(
                    "the exact edited proof was accepted by the operator-authorized "
                    "pinned Lean checker"
                    if verified
                    else "the edited proof has no completed authorized verification"
                ),
                verification_record_uri=(
                    checked.result.verification_record_uri if verified else None
                ),
            ),
            artifact_uris=(
                checked.claim_uri,
                baseline.candidate_uri,
                baseline.certificate_uri,
                *(
                    (baseline.result.verification_record_uri,)
                    if baseline.result.verification_record_uri is not None
                    else ()
                ),
                checked.candidate_uri,
                checked.certificate_uri,
                *(
                    (checked.result.verification_record_uri,)
                    if checked.result.verification_record_uri is not None
                    else ()
                ),
                artifact.artifact_uri,
            ),
        )


def _is_verified(result: ResultEnvelope) -> bool:
    return (
        result.execution.status is ExecutionStatus.COMPLETED
        and result.conclusion is Conclusion.TRUE
        and result.assurance.verification is Verification.VERIFIED
        and result.verification_record_uri is not None
    )
