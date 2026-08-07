"""Standalone, read-only inspection of an immutable Lean proof state.

``lean.proof_state.inspect`` loads an existing immutable proof-state artifact
without mutating or replaying it and returns the structured goals and context
bound to that artifact. No Lean process is started: the returned fields are
exactly those recorded on the immutable artifact, so inspection is available
whenever the artifact is available, regardless of whether the pinned Lean
runtime is installed.
"""

from __future__ import annotations

import time

from pydantic import ValidationError

from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityMode,
    CapabilityProviderRuntime,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.lean_exploration import LeanProofStateArtifact
from jacobian.contracts.lean_proof_state_inspect import (
    LeanProofStateInspectOutput,
    LeanProofStateInspectRequest,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.lean_frontend.artifacts import (
    _environment_digest,
    _source_digest,
    _state_digest_payload,
)
from jacobian.lean_frontend.exploration import _Resources, _runtime_ms
from jacobian.references import LeanCheckerInstallation
from jacobian.storage.errors import StorageError


class LeanProofStateInspectAdapter:
    def __init__(
        self,
        resources: _Resources,
        provider_runtime: CapabilityProviderRuntime,
    ) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.proof_state.inspect",
            version="1",
            title="Inspect an immutable Lean proof state without replay",
            description=(
                "Load an existing immutable replayable Lean proof-state "
                "artifact and return its structured goals, statement, tactic "
                "prefix, and environment bindings. Performs no Lean process "
                "interaction and never mutates or replays the state."
            ),
            provider="jacobian.lean4",
            provider_runtime=provider_runtime,
            modes=(CapabilityMode.EXPLORE,),
            input_schema=LeanProofStateInspectRequest.model_json_schema(),
            output_schema=LeanProofStateInspectOutput.model_json_schema(),
            tags=("lean", "proof-state", "inspection", "exploration"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = LeanProofStateInspectRequest.model_validate(request.input)
        except ValidationError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_LEAN_PROOF_STATE_INSPECT_REQUEST",
                    stage="request_validation",
                    message="The Lean proof-state inspection request is invalid.",
                    hint="Supply a state_uri returned by a proof-state capability.",
                )
            ) from exc
        started = time.monotonic()
        installation: LeanCheckerInstallation = self.resources.installations[
            validated.environment
        ]
        environment_digest = _environment_digest(
            validated.environment,
            installation,
        )
        state = self._load_state(
            validated.state_uri,
            expected_environment=validated.environment,
            expected_environment_digest=environment_digest,
        )
        output = LeanProofStateInspectOutput(
            state_uri=validated.state_uri,
            environment=state.environment,
            environment_digest=state.environment_digest,
            source_digest=state.source_digest,
            state_digest=state.state_digest,
            statement=state.statement,
            tactic_prefix=state.tactic_prefix,
            normalized_goals=state.normalized_goals,
            goal_count=len(state.normalized_goals),
            completed=state.completed,
            imports=state.imports,
            lean_version=state.lean_version,
            lean_commit=state.lean_commit,
            mathlib_commit=state.mathlib_commit,
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=_runtime_ms(started),
                detail="read-only inspection; no Lean process was started",
            ),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="one immutable proof state inspected without replay",
                parameters={
                    "environment": validated.environment.value,
                    "state_uri": validated.state_uri,
                    "state_digest": state.state_digest,
                },
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.COMPLETE,
                basis=(
                    "the artifact's recorded goals are returned in full; "
                    "no search or replay is performed"
                ),
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis=(
                    "inspection returns the immutable artifact's recorded "
                    "fields; it is not a theorem-verification record"
                ),
            ),
            artifact_uris=(validated.state_uri,),
        )

    def _load_state(
        self,
        state_uri: str,
        *,
        expected_environment: LeanEnvironment,
        expected_environment_digest: str,
    ) -> LeanProofStateArtifact:
        try:
            stored = self.resources.store.get(state_uri)
            if (
                stored.manifest.schema_uri != self.resources.state_schema_uri
                or stored.manifest.semantics_uri != self.resources.semantics_uri
            ):
                raise ValueError("artifact is not a Lean proof state")
            state = LeanProofStateArtifact.model_validate(stored.payload)
        except (StorageError, ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_LEAN_PROOF_STATE",
                    stage="state_loading",
                    message="The supplied state artifact is unavailable or invalid.",
                    hint="Use a state URI returned by a proof-state capability.",
                )
            ) from exc
        if (
            state.environment is not expected_environment
            or state.environment_digest != expected_environment_digest
            or state.source_digest
            != _source_digest(state.statement, state.tactic_prefix)
            or state.state_digest != _state_digest_payload(state)
        ):
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="STALE_LEAN_PROOF_STATE",
                    stage="state_validation",
                    message=(
                        "The proof state no longer matches its source or the "
                        "current pinned Lean environment."
                    ),
                    hint="Recreate the proof state under the current environment.",
                )
            )
        return state


def install_lean_proof_state_inspect_capability(
    resources: _Resources,
    provider_runtime: CapabilityProviderRuntime,
) -> LeanProofStateInspectAdapter:
    return LeanProofStateInspectAdapter(resources, provider_runtime)


__all__ = [
    "LeanProofStateInspectAdapter",
    "install_lean_proof_state_inspect_capability",
]
