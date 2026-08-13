"""Validation of immutable proof states against a caller-selected Lean profile."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from pydantic import ValidationError

from jacobian.checker_authorization import LeanCheckerInstallation
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.lean_exploration import LeanProofStateArtifact
from jacobian.contracts.operations import OperationDiagnostic
from jacobian.lean_frontend.artifacts import (
    _environment_imports,
    _source_digest,
    _state_digest_payload,
)
from jacobian.operation_errors import OperationInvocationError
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository


class _StoredProofStateResources(Protocol):
    @property
    def store(self) -> ArtifactRepository: ...

    @property
    def semantics_uri(self) -> str: ...

    @property
    def state_schema_uri(self) -> str: ...

    @property
    def installations(
        self,
    ) -> Mapping[LeanEnvironment, LeanCheckerInstallation]: ...


def _load_validated_proof_state(
    resources: _StoredProofStateResources,
    state_uri: str,
    *,
    expected_environment: LeanEnvironment,
    expected_environment_digest: str,
    invalid_state_hint: str,
) -> LeanProofStateArtifact:
    """Load one state bound to the profile selected by the current request."""

    try:
        stored = resources.store.get(state_uri)
        if (
            stored.manifest.schema_uri != resources.state_schema_uri
            or stored.manifest.semantics_uri != resources.semantics_uri
        ):
            raise ValueError("artifact is not a Lean proof state")
        state = LeanProofStateArtifact.model_validate(stored.payload)
    except (StorageError, ValidationError, ValueError) as exc:
        raise OperationInvocationError(
            OperationDiagnostic(
                code="INVALID_LEAN_PROOF_STATE",
                stage="state_loading",
                message="The supplied state artifact is unavailable or invalid.",
                hint=invalid_state_hint,
            )
        ) from exc

    installation = resources.installations[expected_environment]
    if (
        state.environment is not expected_environment
        or state.environment_digest != expected_environment_digest
        or state.imports != _environment_imports(expected_environment)
        or state.lean_version != installation.lean_version
        or state.lean_commit != installation.lean_commit
        or state.mathlib_commit != installation.mathlib_commit
        or state.source_digest != _source_digest(state.statement, state.tactic_prefix)
        or state.state_digest != _state_digest_payload(state)
    ):
        raise OperationInvocationError(
            OperationDiagnostic(
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


__all__ = ["_StoredProofStateResources", "_load_validated_proof_state"]
