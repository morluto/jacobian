"""Proof-state capability adapter for bounded Lean exploration."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

import jacobian.lean_frontend.exploration as _exploration_support
from jacobian.capability_adapters import parse_capability_input
from jacobian.capability_errors import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInvocationExample,
    CapabilityRequest,
)
from jacobian.contracts.lean import LeanDiagnosticPhase, LeanDiagnosticSource
from jacobian.contracts.lean_exploration import (
    LeanProofStateArtifact,
    LeanProofStateOutput,
    LeanProofStateRequest,
    LeanProofStateTransitionArtifact,
    LeanProofSuccessorState,
    LeanTypedGoal,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.lean_frontend._state_validation import _load_validated_proof_state
from jacobian.lean_frontend.artifacts import (
    _environment_digest,
    _proof_state_command,
    _source_digest,
    _state_payload,
)
from jacobian.lean_frontend.exploration import (
    _normalized_response_goals,
    _request_validation_diagnostic,
    _Resources,
    _runtime_ms,
    _tactic_diagnostics,
    _validate_source_parts,
)
from jacobian.lean_frontend.repl import _response_errors
from jacobian.lean_frontend.repl_protocol import (
    LeanReplProofStepResponse,
    LeanReplValidatedExecution,
)
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed


@dataclass(frozen=True, slots=True)
class LeanProofStateApplication:
    """Typed transition result before the public capability projection."""

    output: LeanProofStateOutput
    execution: Execution
    artifact_uris: tuple[str, ...]


class LeanProofStateAdapter:
    def __init__(self, resources: _Resources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.proof_state.apply_tactic",
            version="3",
            title="Apply one Lean tactic and inspect resulting goals",
            description=(
                "Apply exactly one tactic in either fresh mode "
                "(statement plus optional proof_prefix) or continuation mode "
                "(state_uri plus tactic). Continuation requests must omit statement "
                "and proof_prefix. A clean Lean process reconstructs the immutable "
                "state and returns typed goals plus a durable successor state. "
                "Tactic failures return stable repair diagnostics, a goal index, "
                "and payload-relative source spans."
            ),
            provider="jacobian.lean4",
            provider_runtime=resources.provider_runtime,
            input_schema=LeanProofStateRequest.model_json_schema(),
            output_schema=LeanProofStateOutput.model_json_schema(),
            tags=(
                "lean",
                "proof-state",
                "tactic",
                "goals",
                "exploration",
                "proof-repair",
                "diagnostics",
                "tactic-error",
                "source-span",
            ),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="close_true_with_trivial",
                    description=(
                        "Apply trivial to a replayable proof state for True; "
                        "a completed transition still requires lean.check."
                    ),
                    input=LeanProofStateRequest.model_validate(
                        {
                            "environment": "CORE",
                            "statement": "True",
                            "tactic": "trivial",
                        }
                    ).model_dump(mode="json"),
                ),
                CapabilityInvocationExample(
                    name="continue_returned_state",
                    description=(
                        "Continue an immutable successor state using only its URI, "
                        "the matching environment, and one tactic."
                    ),
                    input=LeanProofStateRequest.model_validate(
                        {
                            "environment": "CORE",
                            "state_uri": "artifact://sha256/" + "a" * 64,
                            "tactic": "rfl",
                        }
                    ).model_dump(
                        mode="json",
                        exclude={"statement", "proof_prefix"},
                    ),
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def prepare(self, request: CapabilityRequest) -> LeanProofStateRequest:
        try:
            return parse_capability_input(LeanProofStateRequest, request.input)
        except ValidationError as exc:
            raise CapabilityInvocationError(
                _request_validation_diagnostic(
                    exc,
                    code="INVALID_LEAN_TRANSITION_REQUEST",
                    subject="The Lean proof-state transition request is invalid",
                    hint=(
                        "For a fresh state use environment, statement, optional "
                        "proof_prefix, and tactic. For a continuation use only the "
                        "returned state_uri, matching environment, and tactic."
                    ),
                )
            ) from exc

    def _resolve_statement_and_prefix(
        self,
        validated: LeanProofStateRequest,
        environment_digest: str,
    ) -> tuple[str, tuple[str, ...], LeanProofStateArtifact | None]:
        """Resolve the statement and tactic prefix from a fresh or bound state."""

        if validated.state_uri is None:
            if validated.statement is None:
                raise ValueError("validated statement is unexpectedly None")
            return validated.statement, validated.proof_prefix, None
        bound_state = _load_validated_proof_state(
            self.resources,
            validated.state_uri,
            expected_environment=validated.environment,
            expected_environment_digest=environment_digest,
            invalid_state_hint="Use a state URI returned by this capability.",
        )
        if bound_state.completed:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="LEAN_PROOF_STATE_COMPLETED",
                    stage="state_validation",
                    message="The supplied proof state has no remaining goals.",
                    hint=(
                        "Send the complete statement and proof to lean.check; "
                        "no further tactic transition is applicable."
                    ),
                )
            )
        if len(bound_state.tactic_prefix) >= 64:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="LEAN_PROOF_STATE_PREFIX_LIMIT",
                    stage="state_validation",
                    message="The replayable proof state reached the 64-tactic limit.",
                    hint=(
                        "Submit a complete proof to lean.check or begin a new "
                        "bounded exploration."
                    ),
                )
            )
        statement = bound_state.statement
        proof_prefix = bound_state.tactic_prefix
        _validate_source_parts(statement, (*proof_prefix, validated.tactic))
        return statement, proof_prefix, bound_state

    def _execute_tactic_and_extract_goals(
        self,
        validated: LeanProofStateRequest,
        command: str,
    ) -> tuple[LeanReplValidatedExecution, tuple[LeanTypedGoal, ...], bool]:
        """Execute the tactic in a clean process and extract typed successor goals."""

        with tempfile.TemporaryDirectory(prefix="jacobian-lean-proof-state-") as root:
            pickle_path = Path(root) / "proof-state.pickle"
            responses = self.resources.repl.execute_clean(
                command=command,
                tactic=validated.tactic,
                environment=validated.environment,
                pickle_path=pickle_path,
            )
            command_response, validation_response, tactic_response = responses
            reconstruction_errors = (
                *_response_errors(command_response),
                *_response_errors(validation_response),
            )
            if reconstruction_errors:
                raise CapabilityInvocationError(
                    CapabilityDiagnostic(
                        code="LEAN_STATE_RECONSTRUCTION_FAILED",
                        stage="state_reconstruction",
                        message=(
                            "Lean could not reconstruct the bound proof state: "
                            f"{reconstruction_errors[0][:500]}"
                        ),
                        hint=(
                            "Recreate the state from the current pinned environment; "
                            "a reconstruction failure is not a proof conclusion."
                        ),
                    )
                )
            tactic_errors = _response_errors(tactic_response)
            accepted = not tactic_errors
            typed_goals: tuple[LeanTypedGoal, ...] = ()
            if accepted:
                if not isinstance(tactic_response, LeanReplProofStepResponse):
                    raise RuntimeError("Lean REPL returned an invalid tactic response")
                try:
                    typed_goals = _exploration_support._extract_typed_goals(
                        self.resources,
                        pickle_path=pickle_path,
                        request=validated,
                    )
                except _exploration_support.LeanHelperError as exc:
                    raise CapabilityInvocationError(
                        CapabilityDiagnostic(
                            code=exc.code,
                            stage="proof_state_extraction",
                            message=(f"Lean helper reported an error: {exc.code}."),
                            hint=(
                                "Retry with smaller goal/context bounds or verify "
                                "that the pinned proof-state helper is installed."
                            ),
                        )
                    ) from exc
                except RuntimeError as exc:
                    raise CapabilityInvocationError(
                        CapabilityDiagnostic(
                            code="LEAN_PROOF_STATE_EXTRACTION_FAILED",
                            stage="proof_state_extraction",
                            message=(
                                "Lean could not produce the bounded typed successor "
                                "proof state."
                            ),
                            hint=(
                                "Retry with smaller goal/context bounds or verify "
                                "that the pinned proof-state helper is installed."
                            ),
                        )
                    ) from exc
        return responses, typed_goals, accepted

    def invoke(self, validated: LeanProofStateRequest) -> OperationProjection:
        applied = self.apply(validated)
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=Completed(
                value=applied.output,
                runtime_ms=applied.execution.runtime_ms,
                detail=applied.execution.detail,
            ),
            publication=PublishedOperation(
                output=applied.output,
                artifact_uris=applied.artifact_uris,
            ),
        )

    def apply(
        self,
        validated: LeanProofStateRequest,
        *,
        diagnostic_phase: LeanDiagnosticPhase = (LeanDiagnosticPhase.TACTIC_EXECUTION),
        diagnostic_source: LeanDiagnosticSource = LeanDiagnosticSource.TACTIC,
        diagnostic_column_offset: int = 0,
    ) -> LeanProofStateApplication:
        """Apply one already validated transition without re-entering the adapter."""

        try:
            if validated.statement is not None:
                _validate_source_parts(
                    validated.statement,
                    (*validated.proof_prefix, validated.tactic),
                )
            else:
                _validate_source_parts("True", (validated.tactic,))
        except ValueError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_LEAN_TRANSITION_REQUEST",
                    stage="request_validation",
                    message="The Lean statement or tactic sequence is invalid.",
                    hint=(
                        "Use one proposition and bounded tactic bodies without "
                        "commands, imports, declarations, sorry, or run_tac."
                    ),
                )
            ) from exc
        started = time.monotonic()
        installation = self.resources.installations[validated.environment]
        environment_digest = _environment_digest(
            validated.environment,
            installation,
        )
        statement, proof_prefix, bound_state = self._resolve_statement_and_prefix(
            validated, environment_digest
        )
        command = _proof_state_command(
            statement=statement,
            proof_prefix=proof_prefix,
        )
        responses, typed_goals, accepted = self._execute_tactic_and_extract_goals(
            validated, command
        )
        _command_response, validation_response, tactic_response = responses
        replayed_goals = _normalized_response_goals(validation_response)
        if bound_state is not None and replayed_goals != bound_state.normalized_goals:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="STALE_LEAN_PROOF_STATE",
                    stage="state_validation",
                    message=(
                        "The clean replay produced goals different from the "
                        "state artifact."
                    ),
                    hint=(
                        "Recreate the state under the current source and "
                        "environment before applying another tactic."
                    ),
                )
            )
        if bound_state is None:
            input_state_payload = _state_payload(
                environment=validated.environment,
                environment_digest=environment_digest,
                statement=statement,
                tactic_prefix=proof_prefix,
                normalized_goals=replayed_goals,
                installation=installation,
            )
            input_state_artifact = self.resources.artifacts.put(
                schema_uri=self.resources.state_schema_uri,
                semantics_uri=self.resources.semantics_uri,
                payload=input_state_payload.model_dump(mode="json"),
                summary="replayable immutable Lean proof state",
            )
            input_state_uri = input_state_artifact.artifact_uri
            input_state = input_state_payload
        else:
            if validated.state_uri is None:
                raise ValueError("validated state_uri is unexpectedly None")
            input_state_uri = validated.state_uri
            input_state = bound_state

        diagnostics = _tactic_diagnostics(
            responses,
            statement=statement,
            final_phase=diagnostic_phase,
            final_source=diagnostic_source,
            final_column_offset=diagnostic_column_offset,
        )
        successor_states: tuple[LeanProofSuccessorState, ...] = ()
        successor_artifact_uris: tuple[str, ...] = ()
        goals: tuple[str, ...] = ()
        completed = False
        if accepted:
            if not isinstance(tactic_response, LeanReplProofStepResponse):
                raise RuntimeError("Lean REPL returned an invalid tactic response")
            goals = _normalized_response_goals(tactic_response)
            if (tactic_response.proof_status == "Completed") != (len(goals) == 0):
                raise RuntimeError(
                    "Lean REPL returned inconsistent completion and goals"
                )
            successor_payload = _state_payload(
                environment=validated.environment,
                environment_digest=environment_digest,
                statement=statement,
                tactic_prefix=(*proof_prefix, validated.tactic),
                normalized_goals=goals,
                installation=installation,
            )
            successor_artifact = self.resources.artifacts.put(
                schema_uri=self.resources.state_schema_uri,
                semantics_uri=self.resources.semantics_uri,
                payload=successor_payload.model_dump(mode="json"),
                parents=(input_state_uri,),
                summary="successor immutable Lean proof state",
            )
            completed = successor_payload.completed
            successor_states = (
                LeanProofSuccessorState(
                    state_uri=successor_artifact.artifact_uri,
                    state_digest=successor_payload.state_digest,
                    normalized_goals=goals,
                    completed=completed,
                ),
            )
            successor_artifact_uris = (successor_artifact.artifact_uri,)

        replay_source = "\n  ".join((*proof_prefix, validated.tactic))
        transition_source_digest = _source_digest(
            statement,
            (*proof_prefix, validated.tactic),
        )
        messages = tuple(diagnostic.message for diagnostic in diagnostics)
        artifact_payload = LeanProofStateTransitionArtifact(
            environment=validated.environment,
            environment_digest=environment_digest,
            source_digest=transition_source_digest,
            statement=statement,
            proof_prefix=proof_prefix,
            tactic=validated.tactic,
            input_state_uri=input_state_uri,
            input_state_digest=input_state.state_digest,
            replay_source=replay_source,
            goals=goals,
            typed_goals=typed_goals,
            goal_count=len(goals),
            successor_states=successor_states,
            accepted=accepted,
            completed=completed,
            messages=messages,
            diagnostics=diagnostics,
            lean_version=installation.lean_version,
            lean_commit=installation.lean_commit,
            mathlib_commit=installation.mathlib_commit,
        )
        artifact = self.resources.artifacts.put(
            schema_uri=self.resources.transition_schema_uri,
            semantics_uri=self.resources.semantics_uri,
            payload=artifact_payload.model_dump(mode="json"),
            parents=(input_state_uri, *successor_artifact_uris),
            summary=(
                "accepted replayable Lean tactic transition"
                if accepted
                else "rejected replayable Lean tactic transition"
            ),
        )
        output = LeanProofStateOutput(
            **artifact_payload.model_dump(mode="python"),
            transition_uri=artifact.artifact_uri,
        )
        artifact_uris = (
            input_state_uri,
            *successor_artifact_uris,
            artifact.artifact_uri,
        )
        return LeanProofStateApplication(
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=_runtime_ms(started),
                detail=(
                    None
                    if accepted
                    else "Lean rejected the tactic; no successor state was created"
                ),
            ),
            output=output,
            artifact_uris=artifact_uris,
        )


__all__ = ["LeanProofStateAdapter", "LeanProofStateApplication"]
