"""Contracts for replayable exploratory Lean operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator

from jacobian.contracts.common import ArtifactUri, Sha256Digest
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.results import ContractModel

LeanNormalizedGoal = Annotated[str, Field(min_length=1, max_length=20_000)]


class LeanProofStateRequest(ContractModel):
    state_uri: ArtifactUri | None = None
    environment: LeanEnvironment = LeanEnvironment.CORE
    statement: str | None = Field(default=None, min_length=1, max_length=2_000)
    proof_prefix: tuple[str, ...] = Field(default=(), max_length=32)
    tactic: str = Field(min_length=1, max_length=1_000)
    max_goals: StrictInt = Field(default=32, ge=1, le=64)
    max_local_declarations: StrictInt = Field(default=128, ge=1, le=256)
    max_rendered_bytes: StrictInt = Field(default=65_536, ge=1_024, le=262_144)

    @model_validator(mode="after")
    def require_one_state_source_and_bounded_prefix(self) -> Self:
        if any(
            not tactic.strip() or len(tactic) > 1_000 for tactic in self.proof_prefix
        ):
            raise ValueError("proof-prefix tactics must be nonempty and bounded")
        if self.state_uri is None and self.statement is None:
            raise ValueError("statement is required when state_uri is omitted")
        if self.state_uri is not None and (
            self.statement is not None or self.proof_prefix
        ):
            raise ValueError(
                "state_uri cannot be combined with statement or proof_prefix"
            )
        return self


class LeanLocalDeclaration(ContractModel):
    user_name: str = Field(min_length=1, max_length=512)
    binder_info: str = Field(min_length=1, max_length=64)
    type: str = Field(min_length=1, max_length=20_000)
    value: str | None = Field(default=None, min_length=1, max_length=20_000)


class LeanTypedGoal(ContractModel):
    goal_index: StrictInt = Field(ge=0, le=63)
    target_type: str = Field(min_length=1, max_length=20_000)
    local_declarations: tuple[LeanLocalDeclaration, ...] = Field(max_length=256)


class LeanProofStateArtifact(ContractModel):
    state_schema_version: Literal["1"] = "1"
    environment: LeanEnvironment
    environment_digest: Sha256Digest
    source_digest: Sha256Digest
    statement: str
    tactic_prefix: tuple[str, ...] = Field(max_length=64)
    normalized_goals: tuple[LeanNormalizedGoal, ...] = Field(max_length=128)
    state_digest: Sha256Digest
    completed: StrictBool
    imports: tuple[str, ...]
    lean_version: str
    lean_commit: str
    mathlib_commit: str | None = None
    expiry: Literal["IMMUTABLE_NO_EXPIRY"] = "IMMUTABLE_NO_EXPIRY"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

    @model_validator(mode="after")
    def require_completion_shape(self) -> Self:
        if self.completed != (len(self.normalized_goals) == 0):
            raise ValueError("state completion differs from normalized goals")
        if len(set(self.imports)) != len(self.imports):
            raise ValueError("state imports must be unique")
        return self


class LeanTacticDiagnostic(ContractModel):
    severity: Literal["ERROR", "WARNING", "INFO"]
    message: str = Field(min_length=1, max_length=20_000)


class LeanProofSuccessorState(ContractModel):
    state_uri: ArtifactUri
    state_digest: Sha256Digest
    normalized_goals: tuple[LeanNormalizedGoal, ...] = Field(max_length=128)
    completed: StrictBool

    @model_validator(mode="after")
    def bind_completion_to_goals(self) -> Self:
        if self.completed != (not self.normalized_goals):
            raise ValueError("successor completion differs from normalized goals")
        return self


class LeanProofStateTransitionArtifact(ContractModel):
    transition_schema_version: Literal["2"] = "2"
    environment: LeanEnvironment
    environment_digest: Sha256Digest
    source_digest: Sha256Digest
    statement: str
    proof_prefix: tuple[str, ...]
    tactic: str
    input_state_uri: ArtifactUri
    input_state_digest: Sha256Digest
    replay_source: str
    goals: tuple[str, ...]
    typed_goals: tuple[LeanTypedGoal, ...]
    goal_count: int = Field(ge=0)
    successor_states: tuple[LeanProofSuccessorState, ...] = Field(max_length=1)
    accepted: StrictBool
    completed: StrictBool
    messages: tuple[str, ...]
    diagnostics: tuple[LeanTacticDiagnostic, ...]
    lean_version: str
    lean_commit: str
    mathlib_commit: str | None = None
    verification_boundary: Literal["LEAN_CHECK_REQUIRED"] = "LEAN_CHECK_REQUIRED"

    @model_validator(mode="after")
    def require_consistent_goal_summary(self) -> Self:
        if self.goal_count != len(self.goals):
            raise ValueError("goal count differs from returned goals")
        if self.goal_count != len(self.typed_goals):
            raise ValueError("goal count differs from returned typed goals")
        if tuple(goal.goal_index for goal in self.typed_goals) != tuple(
            range(self.goal_count)
        ):
            raise ValueError("typed goal indices must be contiguous")
        if self.accepted:
            if self.completed != (self.goal_count == 0):
                raise ValueError("completion differs from returned goals")
            if len(self.successor_states) != 1:
                raise ValueError("an accepted tactic must return one successor state")
            successor = self.successor_states[0]
            if successor.normalized_goals != self.goals:
                raise ValueError("flattened goals differ from the successor state")
            if self.completed != successor.completed:
                raise ValueError("completion differs from the successor state")
        elif self.successor_states or self.goals or self.typed_goals or self.completed:
            raise ValueError("a rejected tactic cannot return successor state")
        return self


class LeanProofStateOutput(LeanProofStateTransitionArtifact):
    transition_uri: ArtifactUri
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"


class LeanPremiseRetrievalRequest(ContractModel):
    environment: Literal["MATHLIB"] = "MATHLIB"
    statement: str = Field(min_length=1, max_length=2_000)
    proof_prefix: tuple[str, ...] = Field(default=(), max_length=32)
    limit: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def require_bounded_prefix(self) -> Self:
        if any(
            not tactic.strip() or len(tactic) > 1_000 for tactic in self.proof_prefix
        ):
            raise ValueError("proof-prefix tactics must be nonempty and bounded")
        return self


class LeanPremiseCandidate(ContractModel):
    rank: int = Field(ge=1, le=20)
    tactic: str = Field(min_length=1, max_length=2_000)
    declaration_names: tuple[str, ...] = ()
    backend: Literal["mathlib.exact?"] = "mathlib.exact?"
    backend_module: Literal["Mathlib.Tactic"] = "Mathlib.Tactic"
    tactic_replayed: bool
    declaration_name_extraction: Literal["DISPLAY_TEXT_HEURISTIC"] = (
        "DISPLAY_TEXT_HEURISTIC"
    )


class LeanPremiseRetrievalArtifact(ContractModel):
    retrieval_schema_version: Literal["2"] = "2"
    environment: Literal["MATHLIB"] = "MATHLIB"
    statement: str
    proof_prefix: tuple[str, ...]
    candidates: tuple[LeanPremiseCandidate, ...]
    exhaustive: Literal[False] = False
    retrieval_api: Literal["MATHLIB_EXACT_TACTIC"] = "MATHLIB_EXACT_TACTIC"
    api_stability: Literal["EXPERIMENTAL_TACTIC_DIAGNOSTIC"] = (
        "EXPERIMENTAL_TACTIC_DIAGNOSTIC"
    )
    goal_context_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    lean_version: str
    lean_commit: str
    mathlib_commit: str


class LeanPremiseRetrievalOutput(LeanPremiseRetrievalArtifact):
    retrieval_uri: ArtifactUri
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"
