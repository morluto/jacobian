"""Premise-retrieval capability adapter for bounded Lean exploration."""

from __future__ import annotations

import hashlib
import time

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.lean_exploration import (
    LeanPremiseCandidate,
    LeanPremiseRetrievalArtifact,
    LeanPremiseRetrievalOutput,
    LeanPremiseRetrievalRequest,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.lean_frontend.artifacts import _proof_state_command
from jacobian.lean_frontend.exploration import (
    _DECLARATION,
    _SUGGESTION,
    _Resources,
    _response_messages,
    _run_repl,
    _runtime_ms,
    _validate_source_parts,
)
from jacobian.lean_frontend.repl import _response_errors
from jacobian.lean_frontend.repl_protocol import LeanReplProofStepResponse


class LeanPremiseRetrievalAdapter:
    def __init__(self, resources: _Resources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.retrieve.premises",
            version="2",
            title="Retrieve Lean premises",
            description=(
                "Ask pinned Mathlib exact? for bounded candidate tactics at one "
                "explicit proof prefix; an empty result is non-exhaustive."
            ),
            provider="jacobian.lean4",
            provider_runtime=resources.provider_runtime,
            input_schema=LeanPremiseRetrievalRequest.model_json_schema(),
            output_schema=LeanPremiseRetrievalOutput.model_json_schema(),
            tags=("lean", "mathlib", "premise-retrieval", "exploration"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = LeanPremiseRetrievalRequest.model_validate(request.input)
            _validate_source_parts(validated.statement, validated.proof_prefix)
        except (ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_LEAN_RETRIEVAL_REQUEST",
                    stage="request_validation",
                    message="The Lean premise-retrieval request is invalid.",
                )
            ) from exc
        started = time.monotonic()
        environment = LeanEnvironment.MATHLIB
        installation = self.resources.installations[environment]
        command = _proof_state_command(
            statement=validated.statement,
            proof_prefix=validated.proof_prefix,
        )
        command_response, tactic_response = _run_repl(
            self.resources,
            command=command,
            tactic="exact?",
            environment=environment,
        )
        command_errors = _response_errors(command_response)
        tactic_errors = _response_errors(tactic_response)
        if command_errors:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="LEAN_RETRIEVAL_FAILED",
                    stage="premise_retrieval",
                    message=(
                        "Lean rejected the statement or proof prefix: "
                        f"{command_errors[0][:500]}"
                    ),
                    hint="Correct the statement or proof prefix and retry.",
                )
            )
        diagnostics = "\n".join(_response_messages(tactic_response))
        suggestions = [
            match.group("tactic").strip() for match in _SUGGESTION.finditer(diagnostics)
        ][: validated.limit]
        if tactic_errors and not any(
            "`exact?` could not close the goal" in error for error in tactic_errors
        ):
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="LEAN_RETRIEVAL_FAILED",
                    stage="premise_retrieval",
                    message=f"Mathlib exact? failed: {tactic_errors[0][:500]}",
                    hint="Correct the statement or proof prefix and retry.",
                )
            )
        candidates = tuple(
            LeanPremiseCandidate(
                rank=index,
                tactic=suggestion,
                declaration_names=tuple(sorted(set(_DECLARATION.findall(suggestion)))),
                tactic_replayed=(
                    index == 1
                    and isinstance(tactic_response, LeanReplProofStepResponse)
                    and tactic_response.proof_status == "Completed"
                ),
            )
            for index, suggestion in enumerate(suggestions, start=1)
        )
        artifact_payload = LeanPremiseRetrievalArtifact(
            statement=validated.statement,
            proof_prefix=validated.proof_prefix,
            candidates=candidates,
            goal_context_digest=(
                "sha256:"
                + hashlib.sha256(
                    canonicalize_json(
                        {
                            "environment": "MATHLIB",
                            "statement": validated.statement,
                            "proof_prefix": list(validated.proof_prefix),
                        }
                    )
                ).hexdigest()
            ),
            lean_version=installation.lean_version,
            lean_commit=installation.lean_commit,
            mathlib_commit=installation.mathlib_commit or "",
        )
        artifact = self.resources.artifacts.put(
            schema_uri=self.resources.retrieval_schema_uri,
            semantics_uri=self.resources.semantics_uri,
            payload=artifact_payload.model_dump(mode="json"),
            summary="non-exhaustive pinned Mathlib premise suggestions",
        )
        output = LeanPremiseRetrievalOutput(
            **artifact_payload.model_dump(mode="python"),
            retrieval_uri=artifact.artifact_uri,
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=_runtime_ms(started),
            ),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="one explicit Lean goal under pinned Mathlib exact?",
                parameters={
                    "environment": "MATHLIB",
                    "statement": validated.statement,
                    "limit": validated.limit,
                },
                artifact_uri=artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.PARTIAL,
                basis=(
                    "Mathlib exact? suggestions are bounded and non-exhaustive; "
                    "no suggestion is not a proof of absence"
                ),
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis=(
                    "candidate tactics were emitted by pinned Mathlib exact?; "
                    "they remain unverified until lean.check accepts exact source"
                ),
            ),
            artifact_uris=(artifact.artifact_uri,),
        )


__all__ = ["LeanPremiseRetrievalAdapter"]
