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
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.checker_authorization import LeanCheckerInstallation
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.lean_proof_state_inspect import (
    LeanProofStateInspectOutput,
    LeanProofStateInspectRequest,
)
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiagnostic,
    OperationInputKind,
    OperationRequest,
    ProviderObservation,
)
from jacobian.lean_frontend._state_validation import (
    _load_validated_proof_state,
    _StoredProofStateResources,
)
from jacobian.lean_frontend.artifacts import _environment_digest
from jacobian.operation_adapters import parse_operation_input
from jacobian.operation_errors import OperationInvocationError
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository


@dataclass(frozen=True, slots=True)
class _InspectionResources:
    store: ArtifactRepository
    semantics_uri: str
    state_schema_uri: str
    installations: Mapping[LeanEnvironment, LeanCheckerInstallation]


class LeanProofStateInspectAdapter:
    def __init__(
        self,
        resources: _StoredProofStateResources,
        provider_runtime: ProviderObservation,
    ) -> None:
        self.resources = resources
        self._descriptor = OperationDescriptor(
            operation_id="lean.proof_state.inspect",
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
            input_schema=LeanProofStateInspectRequest.model_json_schema(),
            output_schema=LeanProofStateInspectOutput.model_json_schema(),
            read_only=True,
            tags=("lean", "proof-state", "inspection", "exploration"),
            accepted_input_kinds=(
                OperationInputKind.STRUCTURED_REQUEST,
                OperationInputKind.TYPED_ARTIFACT,
            ),
            accepted_artifact_types=(resources.state_schema_uri,),
        )

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> LeanProofStateInspectRequest:
        try:
            validated = parse_operation_input(
                LeanProofStateInspectRequest, request.input
            )
        except ValidationError as exc:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="INVALID_LEAN_PROOF_STATE_INSPECT_REQUEST",
                    stage="request_validation",
                    message="The Lean proof-state inspection request is invalid.",
                    hint="Supply a state_uri returned by a proof-state operation.",
                )
            ) from exc
        return validated

    def invoke(self, validated: LeanProofStateInspectRequest) -> OperationProjection:
        started = time.monotonic()
        installation: LeanCheckerInstallation = self.resources.installations[
            validated.environment
        ]
        environment_digest = _environment_digest(
            validated.environment,
            installation,
        )
        state = _load_validated_proof_state(
            self.resources,
            validated.state_uri,
            expected_environment=validated.environment,
            expected_environment_digest=environment_digest,
            invalid_state_hint=("Use a state URI returned by a proof-state operation."),
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
        return OperationProjection(
            operation_id=self.descriptor.operation_id,
            version=self.descriptor.version,
            terminal=Completed(
                value=output,
                runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
                detail="read-only inspection; no Lean process was started",
            ),
            publication=PublishedOperation(
                output=output,
                artifact_uris=(validated.state_uri,),
            ),
        )


def install_lean_proof_state_inspect_operation(
    resources: _StoredProofStateResources,
    provider_runtime: ProviderObservation,
) -> LeanProofStateInspectAdapter:
    return LeanProofStateInspectAdapter(resources, provider_runtime)


def install_lean_proof_state_inspect_only(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    installations: Mapping[LeanEnvironment, LeanCheckerInstallation],
    provider_runtime: ProviderObservation,
) -> LeanProofStateInspectAdapter:
    """Register the read-only proof-state inspect adapter without a Lean runtime.

    ``lean.proof_state.inspect`` only reads stored artifacts and never starts a
    Lean process, so it remains available when the pinned Lean runtime is
    absent. The semantics and state schema registered here are reused by the
    full exploration installer when the runtime is available.
    """

    from jacobian.contracts.lean_exploration import LeanProofStateArtifact

    del artifacts

    core = installations[LeanEnvironment.CORE]
    mathlib = installations[LeanEnvironment.MATHLIB]
    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.lean4-exploration",
        version="1",
        definition={
            "description": (
                "immutable replayable Lean proof states, one-step tactic "
                "transitions, and premise suggestions"
            ),
            "lean_version": core.lean_version,
            "lean_commit": core.lean_commit,
            "mathlib_commit": mathlib.mathlib_commit,
            "state_expiry": "immutable artifacts do not expire",
            "verification": "none; completed source must pass lean.check",
        },
    )
    state_schema_uri = schemas.register(
        name="jacobian.lean4-proof-state",
        version="1",
        schema=LeanProofStateArtifact.model_json_schema(),
    )
    resources = _InspectionResources(
        store=store,
        semantics_uri=semantics_uri,
        state_schema_uri=state_schema_uri,
        installations=installations,
    )
    return LeanProofStateInspectAdapter(resources, provider_runtime)


__all__ = [
    "LeanProofStateInspectAdapter",
    "install_lean_proof_state_inspect_only",
    "install_lean_proof_state_inspect_operation",
]
