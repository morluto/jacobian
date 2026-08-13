"""Premise-retrieval capability adapter for bounded Lean exploration."""

from __future__ import annotations

import hashlib
import time

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
from jacobian.capability_adapters import parse_capability_input
from jacobian.capability_errors import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInvocationExample,
    CapabilityRequest,
)
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.lean_exploration import (
    LeanPremiseCandidate,
    LeanPremiseRetrievalArtifact,
    LeanPremiseRetrievalOutput,
    LeanPremiseRetrievalRequest,
)
from jacobian.lean_frontend.artifacts import _proof_state_command
from jacobian.lean_frontend.exploration import (
    _DECLARATION,
    _SUGGESTION,
    _request_validation_diagnostic,
    _Resources,
    _response_messages,
    _run_repl,
    _runtime_ms,
    _validate_source_parts,
)
from jacobian.lean_frontend.repl import _response_errors
from jacobian.lean_frontend.repl_protocol import LeanReplProofStepResponse
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed


class LeanPremiseRetrievalAdapter:
    def __init__(self, resources: _Resources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.retrieve.premises",
            version="2",
            title="Retrieve Lean premises",
            description=(
                "Retrieve bounded premise-backed tactic candidates for one explicit "
                "statement and tactic-body prefix with pinned Mathlib's `exact?` "
                "tactic. "
                "proof_prefix contains only tactics after `by` (for example "
                "[`intro x`]); never include `by`. Each candidate reports whether "
                "the first tactic replayed, and an empty result is non-exhaustive."
            ),
            provider="jacobian.lean4",
            provider_runtime=resources.provider_runtime,
            input_schema=LeanPremiseRetrievalRequest.model_json_schema(),
            output_schema=LeanPremiseRetrievalOutput.model_json_schema(),
            tags=("lean", "mathlib", "premise-retrieval", "exploration"),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="square_nonnegative_after_intro",
                    description=(
                        "Ask Mathlib exact? for a premise after introducing x; "
                        "proof_prefix omits the surrounding `by`."
                    ),
                    input=LeanPremiseRetrievalRequest.model_validate(
                        {
                            "statement": "∀ x : Real, x ^ 2 ≥ 0",
                            "proof_prefix": ["intro x"],
                            "limit": 5,
                        }
                    ).model_dump(mode="json"),
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def prepare(self, request: CapabilityRequest) -> LeanPremiseRetrievalRequest:
        try:
            validated = parse_capability_input(
                LeanPremiseRetrievalRequest, request.input
            )
            _validate_source_parts(validated.statement, validated.proof_prefix)
        except (ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                _request_validation_diagnostic(
                    exc,
                    code="INVALID_LEAN_RETRIEVAL_REQUEST",
                    subject="The Lean premise-retrieval request is invalid",
                    hint=(
                        "Use one proposition and a bounded sequence of tactic bodies "
                        "after `by`; do not include the `by` introducer."
                    ),
                )
            ) from exc
        return validated

    def invoke(self, validated: LeanPremiseRetrievalRequest) -> OperationProjection:
        started = time.monotonic()
        environment = LeanEnvironment.MATHLIB
        installation = self.resources.installations[environment]
        command = _proof_state_command(
            statement=validated.statement,
            proof_prefix=validated.proof_prefix,
        )
        try:
            command_response, tactic_response = _run_repl(
                self.resources,
                command=command,
                tactic="exact?",
                environment=environment,
            )
        except RuntimeError as exc:
            raw_backend_message = str(exc)[:1_000]
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="LEAN_RETRIEVAL_FAILED",
                    stage="premise_retrieval",
                    message="Pinned Mathlib premise retrieval did not return a result.",
                    hint=(
                        "Inspect raw_backend_message, correct the statement or tactic "
                        "prefix when applicable, and retry."
                    ),
                    details={"raw_backend_message": raw_backend_message},
                )
            ) from exc
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
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=Completed(value=output, runtime_ms=_runtime_ms(started)),
            publication=PublishedOperation(
                output=output,
                artifact_uris=(artifact.artifact_uri,),
            ),
        )


__all__ = ["LeanPremiseRetrievalAdapter"]
