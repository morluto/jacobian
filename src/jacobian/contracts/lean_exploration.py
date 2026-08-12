"""Contracts for replayable exploratory Lean operations."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, StrictBool, StrictInt, model_validator

from jacobian.contracts.common import ArtifactUri, Sha256Digest
from jacobian.contracts.lean import (
    LeanDiagnostic,
    LeanEnvironment,
)
from jacobian.contracts.results import ContractModel

LeanNormalizedGoal = Annotated[str, Field(min_length=1, max_length=20_000)]


def _require_tactic_body(value: str) -> str:
    if not value.strip():
        raise ValueError("tactic bodies must be nonempty")
    if re.match(r"^by(?:\s|$)", value.lstrip()):
        raise ValueError(
            "tactic bodies must not include `by`, the surrounding proof introducer"
        )
    return value


LeanTacticBody = Annotated[
    str,
    Field(min_length=1, max_length=1_000),
    AfterValidator(_require_tactic_body),
]


class LeanProofStateRequest(ContractModel):
    state_uri: ArtifactUri | None = Field(
        default=None,
        description=(
            "Continuation mode: an immutable successor state URI returned by an "
            "earlier tactic transition. When supplied, omit statement and proof_prefix."
        ),
    )
    environment: LeanEnvironment = Field(
        default=LeanEnvironment.CORE,
        description=(
            "Pinned Lean environment. It must match the bound state in continuation "
            "mode."
        ),
    )
    statement: str | None = Field(
        default=None,
        min_length=1,
        max_length=2_000,
        description=(
            "Fresh mode: one proposition expression to prove. Required when state_uri "
            "is omitted and forbidden in continuation mode."
        ),
    )
    proof_prefix: tuple[LeanTacticBody, ...] = Field(
        default=(),
        max_length=32,
        description=(
            "Fresh-mode tactic bodies already applied after Lean's `by`. Do not "
            "include `by`; omit this field in continuation mode."
        ),
    )
    tactic: LeanTacticBody = Field(
        description="Exactly one next tactic body to apply; do not include `by`."
    )
    max_goals: StrictInt = Field(
        default=32,
        ge=1,
        le=64,
        description="Maximum successor goals returned by the typed state extractor.",
    )
    max_local_declarations: StrictInt = Field(
        default=128,
        ge=1,
        le=256,
        description="Maximum local declarations returned across one typed goal.",
    )
    max_rendered_bytes: StrictInt = Field(
        default=65_536,
        ge=1_024,
        le=262_144,
        description="Maximum rendered bytes returned by typed goal extraction.",
    )

    @model_validator(mode="after")
    def require_one_state_source_and_bounded_prefix(self) -> Self:
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

    @model_validator(mode="after")
    def require_completion_shape(self) -> Self:
        if self.completed != (len(self.normalized_goals) == 0):
            raise ValueError("state completion differs from normalized goals")
        if len(set(self.imports)) != len(self.imports):
            raise ValueError("state imports must be unique")
        return self


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
    transition_schema_version: Literal["3"] = "3"
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
    diagnostics: tuple[LeanDiagnostic, ...]
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


class LeanPremiseRetrievalRequest(ContractModel):
    environment: Literal["MATHLIB"] = Field(
        default="MATHLIB",
        description="Premise retrieval is available only in the pinned MATHLIB profile.",
    )
    statement: str = Field(
        min_length=1,
        max_length=2_000,
        description="One Lean proposition expression whose current goal needs a premise.",
    )
    proof_prefix: tuple[LeanTacticBody, ...] = Field(
        default=(),
        max_length=32,
        description=(
            "Tactic bodies already applied after Lean's `by`, for example "
            "[`intro x`]. Do not include `by`."
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum non-exhaustive Mathlib exact? candidates to return.",
    )


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
