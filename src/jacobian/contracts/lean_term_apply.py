"""Contracts for bounded Lean term application to a replayable proof state.

``lean.term.apply`` applies one bounded Lean term to an immutable proof-state
artifact by elaborating ``exact <term>`` through the maintained Lean REPL
tactic protocol. It reuses the existing replayable proof-state transition
artifact types from :mod:`jacobian.contracts.lean_exploration` so that term
application and tactic application share one typed successor-state shape and
one verification boundary.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.common import ArtifactUri
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.lean_exploration import LeanProofStateOutput
from jacobian.contracts.results import ContractModel


class LeanTermApplyRequest(ContractModel):
    state_uri: ArtifactUri | None = None
    environment: LeanEnvironment = LeanEnvironment.CORE
    statement: str | None = Field(default=None, min_length=1, max_length=2_000)
    proof_prefix: tuple[str, ...] = Field(default=(), max_length=32)
    term: str = Field(min_length=1, max_length=994)
    max_goals: StrictInt = Field(default=32, ge=1, le=64)
    max_local_declarations: StrictInt = Field(default=128, ge=1, le=256)
    max_rendered_bytes: StrictInt = Field(default=65_536, ge=1_024, le=262_144)

    @model_validator(mode="after")
    def require_one_state_source_and_bounded_term(self) -> Self:
        if "\n" in self.term or "\r" in self.term or ":=" in self.term:
            raise ValueError("term must be one Lean expression")
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


class LeanTermApplyOutput(LeanProofStateOutput):
    """Term-application output shares the transition artifact shape.

    The ``tactic`` field of the inherited transition artifact holds the
    elaborated ``exact <term>`` string so that the replay source remains
    exact and inspectable. ``term_apply_uri`` identifies this transition.
    """

    term_apply_uri: ArtifactUri
    term_application: Literal["LEAN_EXACT_ELABORATION"] = "LEAN_EXACT_ELABORATION"


__all__ = ["LeanTermApplyOutput", "LeanTermApplyRequest"]
