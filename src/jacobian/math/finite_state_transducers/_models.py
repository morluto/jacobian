"""Typed wire contracts for finite-state transducer operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.finite_state_transducers.values import (
    MAX_FST_ALPHABET,
    MAX_FST_STATES,
    MAX_FST_WORD_LENGTH,
    RationalTransducer,
    SubsequentialTransducer,
)

# ---------------------------------------------------------------------------
# Subsequential run
# ---------------------------------------------------------------------------


class SubseqRunRequest(StrictModel):
    """Run a subsequential transducer on an input word."""

    transducer: SubsequentialTransducer
    word: tuple[int, ...] = Field(max_length=MAX_FST_WORD_LENGTH)

    @model_validator(mode="after")
    def require_valid_word(self) -> Self:
        for symbol in self.word:
            if not 0 <= symbol < self.transducer.input_alphabet_size:
                raise ValueError("word symbols must be in 0..input_alphabet_size-1")
        return self


class SubseqRunResult(StrictModel):
    """Result of a subsequential run.

    ``status`` is ``OUTPUT`` when the function is defined on the word;
    ``UNDEFINED_TRANSITION`` when a transition is missing;
    ``NONFINAL_DOMAIN_STATE`` when the final state lacks a final output.
    """

    status: Literal["OUTPUT", "UNDEFINED_TRANSITION", "NONFINAL_DOMAIN_STATE"]
    output: tuple[int, ...] = Field(default=())
    final_state: int = Field(ge=0, le=MAX_FST_STATES - 1)
    undefined_position: int | None = None
    partial_output: tuple[int, ...] = Field(default=())


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class IdentityRequest(StrictModel):
    """Construct the identity subsequential transducer on one alphabet."""

    alphabet_size: int = Field(ge=1, le=MAX_FST_ALPHABET)


class IdentityResult(StrictModel):
    """The identity transducer."""

    transducer: SubsequentialTransducer


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class ComposeRequest(StrictModel):
    """Compose T: A* -> B* followed by U: B* -> C*."""

    first: SubsequentialTransducer
    second: SubsequentialTransducer

    @model_validator(mode="after")
    def require_compatible_alphabets(self) -> Self:
        if self.first.output_alphabet_size != self.second.input_alphabet_size:
            raise ValueError(
                "first output alphabet must match second input alphabet"
            )
        return self


class ComposeResult(StrictModel):
    """The composite subsequential transducer."""

    transducer: SubsequentialTransducer


# ---------------------------------------------------------------------------
# Trim
# ---------------------------------------------------------------------------


class TrimRequest(StrictModel):
    """Trim a subsequential transducer to reachable + coaccessible states."""

    transducer: SubsequentialTransducer


class TrimResult(StrictModel):
    """The trimmed transducer and state mapping."""

    transducer: SubsequentialTransducer
    state_map: dict[int, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rational relation inverse
# ---------------------------------------------------------------------------


class RelationInverseRequest(StrictModel):
    """Invert a rational transducer."""

    transducer: RationalTransducer


class RelationInverseResult(StrictModel):
    """The inverted transducer."""

    transducer: RationalTransducer


# ---------------------------------------------------------------------------
# Rational relation path replay
# ---------------------------------------------------------------------------


class RelationPathReplayRequest(StrictModel):
    """Replay a candidate edge path and check it is an valid accepting path."""

    transducer: RationalTransducer
    edge_path: tuple[int, ...] = Field(max_length=MAX_FST_WORD_LENGTH)


class RelationPathReplayResult(StrictModel):
    """Result of replaying one edge path."""

    status: Literal["ACCEPTING_PAIR", "INVALID_PATH"]
    input_word: tuple[int, ...] = Field(default=())
    output_word: tuple[int, ...] = Field(default=())
    state_trace: tuple[int, ...] = Field(default=())
    error: str | None = None


__all__ = [
    "ComposeRequest",
    "ComposeResult",
    "IdentityRequest",
    "IdentityResult",
    "RelationInverseRequest",
    "RelationInverseResult",
    "RelationPathReplayRequest",
    "RelationPathReplayResult",
    "SubseqRunRequest",
    "SubseqRunResult",
    "TrimRequest",
    "TrimResult",
]
