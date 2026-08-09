"""Structured metavariable, local-instance, and elaboration fields for a proof state.

``lean.proof_state.metavariable_fields`` reconstructs an immutable proof-state
artifact in a clean pinned Lean process, pickles it, and asks the pinned
helper to expose typed fields from ``MetavarDecl``, ``LocalInstances``, and
the elaboration ``Term.Context`` through maintained Lean accessors. Coercion
provenance is reported honestly as ``UNAVAILABLE`` because the maintained
``Lean.Meta.Coe`` APIs operate on expressions during elaboration and retain no
per-metavariable coercion log on a pickled state.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from pydantic import ValidationError

import jacobian.lean_frontend.exploration as _exploration_support
from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInputKind,
    CapabilityMode,
    CapabilityProviderRuntime,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.lean_metavariable_fields import (
    LeanMetavariableFieldsArtifact,
    LeanMetavariableFieldsOutput,
    LeanMetavariableFieldsRequest,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.lean_frontend._state_validation import _load_validated_proof_state
from jacobian.lean_frontend.artifacts import (
    _environment_digest,
    _proof_state_command,
    _source_digest,
)
from jacobian.lean_frontend.exploration import (
    _Resources,
    _runtime_ms,
    _validate_source_parts,
)
from jacobian.lean_frontend.repl import _response_errors


class LeanMetavariableFieldsAdapter:
    def __init__(
        self,
        resources: _Resources,
        metavariable_schema_uri: str,
        provider_runtime: CapabilityProviderRuntime,
    ) -> None:
        self.resources = resources
        self.metavariable_schema_uri = metavariable_schema_uri
        self._descriptor = CapabilityDescriptor(
            capability_id="lean.proof_state.metavariable_fields",
            version="1",
            title="Expose structured Lean metavariable and elaboration fields",
            description=(
                "Reconstruct an immutable proof state in a clean pinned Lean "
                "process and expose typed metavariable, local-instance, and "
                "elaboration-context fields through maintained Lean.Meta "
                "accessors. Coercion provenance is reported as UNAVAILABLE."
            ),
            provider="jacobian.lean4",
            provider_runtime=provider_runtime,
            modes=(CapabilityMode.EXPLORE,),
            input_schema=LeanMetavariableFieldsRequest.model_json_schema(),
            output_schema=LeanMetavariableFieldsOutput.model_json_schema(),
            tags=("lean", "proof-state", "metavariable", "exploration"),
            accepted_input_kinds=(
                CapabilityInputKind.STRUCTURED_REQUEST,
                CapabilityInputKind.TYPED_ARTIFACT,
            ),
            accepted_artifact_types=(resources.state_schema_uri,),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = LeanMetavariableFieldsRequest.model_validate(request.input)
        except ValidationError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_LEAN_METAVARIABLE_FIELDS_REQUEST",
                    stage="request_validation",
                    message=("The Lean metavariable-fields request is invalid."),
                    hint="Supply a state_uri returned by a proof-state capability.",
                )
            ) from exc
        started = time.monotonic()
        installation = self.resources.installations[validated.environment]
        environment_digest = _environment_digest(
            validated.environment,
            installation,
        )
        bound_state = _load_validated_proof_state(
            self.resources,
            validated.state_uri,
            expected_environment=validated.environment,
            expected_environment_digest=environment_digest,
            invalid_state_hint=(
                "Use a state URI returned by a proof-state capability."
            ),
        )
        if bound_state.completed:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="LEAN_PROOF_STATE_COMPLETED",
                    stage="state_validation",
                    message="The supplied proof state has no remaining goals.",
                    hint=(
                        "Metavariable fields are only defined for states with "
                        "open goals."
                    ),
                )
            )
        statement = bound_state.statement
        proof_prefix = bound_state.tactic_prefix
        _validate_source_parts(statement, proof_prefix)
        command = _proof_state_command(
            statement=statement,
            proof_prefix=proof_prefix,
        )
        with tempfile.TemporaryDirectory(
            prefix="jacobian-lean-metavariable-fields-"
        ) as root:
            pickle_path = Path(root) / "metavariable-fields.pickle"
            responses = self.resources.repl.execute_clean(
                command=command,
                tactic="skip",
                environment=validated.environment,
                pickle_path=pickle_path,
            )
            command_response, validation_response, skip_response = responses
            reconstruction_errors = (
                *_response_errors(command_response),
                *_response_errors(validation_response),
                *_response_errors(skip_response),
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
                            "Recreate the state from the current pinned "
                            "environment; a reconstruction failure is not a "
                            "proof conclusion."
                        ),
                    )
                )
            try:
                payload = _exploration_support._extract_structured_metavariables(
                    self.resources,
                    pickle_path=pickle_path,
                    request=validated,
                )
            except _exploration_support.LeanHelperError as exc:
                raise CapabilityInvocationError(
                    CapabilityDiagnostic(
                        code=exc.code,
                        stage="metavariable_field_extraction",
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
                        code="LEAN_METAVARIABLE_FIELDS_EXTRACTION_FAILED",
                        stage="metavariable_field_extraction",
                        message=(
                            "Lean could not produce the structured metavariable fields."
                        ),
                        hint=(
                            "Retry with smaller goal/context bounds or verify "
                            "that the pinned proof-state helper is installed."
                        ),
                    )
                ) from exc
        replayed_goals = _exploration_support._normalized_response_goals(
            validation_response
        )
        if replayed_goals != bound_state.normalized_goals:
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
                        "environment before requesting metavariable fields."
                    ),
                )
            )
        structured = payload.structured_metavariables
        if len(structured) != len(bound_state.normalized_goals):
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="LEAN_METAVARIABLE_FIELDS_EXTRACTION_FAILED",
                    stage="metavariable_field_extraction",
                    message=(
                        "The helper returned a metavariable-fields count that "
                        "does not match the state's open goal count."
                    ),
                    hint=(
                        "Recreate the state under the current environment or "
                        "retry with smaller goal/context bounds."
                    ),
                )
            )
        artifact_payload = LeanMetavariableFieldsArtifact(
            environment=validated.environment,
            environment_digest=environment_digest,
            source_digest=_source_digest(statement, proof_prefix),
            state_uri=validated.state_uri,
            state_digest=bound_state.state_digest,
            structured_metavariables=structured,
            elaboration_context=payload.elaboration_context,
            coercion_provenance=payload.coercion_provenance,
            coercion_provenance_basis=payload.coercion_provenance_basis,
            lean_version=installation.lean_version,
            lean_commit=installation.lean_commit,
            mathlib_commit=installation.mathlib_commit,
        )
        artifact = self.resources.artifacts.put(
            schema_uri=self.metavariable_schema_uri,
            semantics_uri=self.resources.semantics_uri,
            payload=artifact_payload.model_dump(mode="json"),
            parents=(validated.state_uri,),
            summary="structured Lean metavariable and elaboration fields",
        )
        output = LeanMetavariableFieldsOutput(
            **artifact_payload.model_dump(mode="python"),
            metavariable_fields_uri=artifact.artifact_uri,
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=_runtime_ms(started),
                detail=None,
            ),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description=(
                    "structured metavariable and elaboration fields for one "
                    "replayed proof state"
                ),
                parameters={
                    "environment": validated.environment.value,
                    "state_uri": validated.state_uri,
                    "state_digest": bound_state.state_digest,
                    "environment_digest": environment_digest,
                },
                artifact_uri=artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.COMPLETE,
                basis=(
                    "the helper reported every open goal's metavariable "
                    "fields for this state"
                ),
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis=(
                    "a clean pinned Lean process reconstructed the bound "
                    "state and exposed maintained MetaM fields; coercion "
                    "provenance is UNAVAILABLE and this is not a "
                    "theorem-verification record"
                ),
            ),
            artifact_uris=(validated.state_uri, artifact.artifact_uri),
        )


def install_lean_metavariable_fields_capability(
    resources: _Resources,
    metavariable_schema_uri: str,
    provider_runtime: CapabilityProviderRuntime,
) -> LeanMetavariableFieldsAdapter:
    return LeanMetavariableFieldsAdapter(
        resources, metavariable_schema_uri, provider_runtime
    )


__all__ = [
    "LeanMetavariableFieldsAdapter",
    "install_lean_metavariable_fields_capability",
]
