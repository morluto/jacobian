"""Contracts for standalone, read-only inspection of an immutable proof state.

``lean.proof_state.inspect`` loads an existing immutable proof-state artifact
without mutating or replaying it and returns the structured goals and context
bound to that artifact. It performs no Lean process interaction: the returned
goals, statement, tactic prefix, and environment bindings are exactly those
recorded on the immutable artifact. A read-only inspection is therefore
always available when the artifact is available, regardless of whether the
pinned Lean runtime is installed.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator

from jacobian.contracts.common import ArtifactUri, Sha256Digest
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.lean_exploration import LeanNormalizedGoal
from jacobian.contracts.results import ContractModel


class LeanProofStateInspectRequest(ContractModel):
    state_uri: ArtifactUri
    environment: LeanEnvironment = LeanEnvironment.CORE


class LeanProofStateInspectOutput(ContractModel):
    inspect_schema_version: Literal["1"] = "1"
    state_uri: ArtifactUri
    environment: LeanEnvironment
    environment_digest: Sha256Digest
    source_digest: Sha256Digest
    state_digest: Sha256Digest
    statement: str
    tactic_prefix: tuple[str, ...] = Field(max_length=64)
    normalized_goals: tuple[LeanNormalizedGoal, ...] = Field(max_length=128)
    goal_count: StrictInt = Field(ge=0, le=128)
    completed: StrictBool
    imports: tuple[str, ...]
    lean_version: str
    lean_commit: str
    mathlib_commit: str | None = None
    inspection: Literal["READ_ONLY_NO_REPLAY"] = "READ_ONLY_NO_REPLAY"

    @model_validator(mode="after")
    def bind_goal_summary(self) -> Self:
        if self.goal_count != len(self.normalized_goals):
            raise ValueError("goal count differs from normalized goals")
        if self.completed != (self.goal_count == 0):
            raise ValueError("completion differs from normalized goals")
        if len(set(self.imports)) != len(self.imports):
            raise ValueError("proof-state imports must be unique")
        return self


__all__ = ["LeanProofStateInspectOutput", "LeanProofStateInspectRequest"]
