"""Typed wire contracts for exact bounded finite-state transducers."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.finite_state_transducers.values import (
    MAX_FST_RESULT_WORD_LENGTH,
    MAX_FST_STATES,
    MAX_FST_WORD_LENGTH,
    RationalTransducer,
    SubsequentialTransducer,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by transducer contracts."""

    return PydanticCustomError(f"finite_state_transducer.{reason}", message)


class SubseqRunRequest(StrictModel):
    transducer: SubsequentialTransducer
    word: tuple[int, ...] = Field(max_length=MAX_FST_WORD_LENGTH)

    @model_validator(mode="after")
    def require_valid_bounded_word(self) -> Self:
        if any(
            not 0 <= symbol < self.transducer.input_alphabet_size
            for symbol in self.word
        ):
            raise _validation_error(
                "word_symbol_out_of_range", "word symbol is outside the input alphabet"
            )
        transition_bound = max(
            (len(transition.output) for transition in self.transducer.transitions),
            default=0,
        )
        final_bound = max(
            (len(final.output) for final in self.transducer.final_outputs), default=0
        )
        if len(self.word) * transition_bound + final_bound > MAX_FST_RESULT_WORD_LENGTH:
            raise _validation_error(
                "run_output_exceeds_bound",
                "subsequential output may exceed the result word bound",
            )
        return self


class SubseqRunResult(SubseqRunRequest):
    status: Literal["OUTPUT", "UNDEFINED_TRANSITION", "NONFINAL_DOMAIN_STATE"]
    output: tuple[int, ...] = Field(max_length=MAX_FST_RESULT_WORD_LENGTH)
    final_state: int = Field(ge=0, lt=MAX_FST_STATES)
    undefined_position: int | None = None
    partial_output: tuple[int, ...] = Field(max_length=MAX_FST_RESULT_WORD_LENGTH)

    @model_validator(mode="after")
    def bind_run(self) -> Self:
        from jacobian.math.finite_state_transducers.operations import run_subsequential

        expected = run_subsequential(self.transducer, self.word)
        actual = (
            self.status,
            self.output,
            self.final_state,
            self.undefined_position,
            self.partial_output,
        )
        if actual != expected:
            raise _validation_error(
                "run_result_mismatch",
                "result must match the exact bound subsequential run",
            )
        return self


class ComposeRequest(StrictModel):
    first: SubsequentialTransducer
    second: SubsequentialTransducer

    @model_validator(mode="after")
    def require_bounded_compatible_composition(self) -> Self:
        if self.first.output_alphabet_size != self.second.input_alphabet_size:
            raise _validation_error(
                "composition_alphabet_mismatch",
                "first output alphabet must match second input alphabet",
            )
        if self.first.state_count * self.second.state_count > MAX_FST_STATES:
            raise _validation_error(
                "composition_state_bound_exceeded",
                "composite product-state bound exceeds 64",
            )
        second_transition_bound = max(
            (len(transition.output) for transition in self.second.transitions),
            default=0,
        )
        first_transition_bound = max(
            (len(transition.output) for transition in self.first.transitions),
            default=0,
        )
        first_final_bound = max(
            (len(final.output) for final in self.first.final_outputs), default=0
        )
        second_final_bound = max(
            (len(final.output) for final in self.second.final_outputs), default=0
        )
        if first_transition_bound * second_transition_bound > MAX_FST_WORD_LENGTH:
            raise _validation_error(
                "composition_transition_output_exceeds_bound",
                "composite transition output may exceed the word bound",
            )
        if (
            first_final_bound * second_transition_bound + second_final_bound
            > MAX_FST_WORD_LENGTH
        ):
            raise _validation_error(
                "composition_final_output_exceeds_bound",
                "composite final output may exceed the word bound",
            )
        return self


class ComposeResult(ComposeRequest):
    transducer: SubsequentialTransducer

    @model_validator(mode="after")
    def bind_composition(self) -> Self:
        from jacobian.math.finite_state_transducers.operations import (
            compose_subsequential,
        )

        if self.transducer != compose_subsequential(self.first, self.second):
            raise _validation_error(
                "composition_result_mismatch",
                "transducer must be the exact bound composition",
            )
        return self


class RelationPathReplayRequest(StrictModel):
    transducer: RationalTransducer
    initial_state: int = Field(ge=0, lt=MAX_FST_STATES)
    edge_path: tuple[int, ...] = Field(max_length=MAX_FST_WORD_LENGTH)

    @model_validator(mode="after")
    def require_selected_start_and_bounded_labels(self) -> Self:
        if self.initial_state not in self.transducer.initial_states:
            raise _validation_error(
                "initial_state_not_declared",
                "initial_state must select one declared initial state",
            )
        if all(0 <= index < len(self.transducer.edges) for index in self.edge_path):
            input_length = sum(
                len(self.transducer.edges[index].input_label)
                for index in self.edge_path
            )
            output_length = sum(
                len(self.transducer.edges[index].output_label)
                for index in self.edge_path
            )
            if max(input_length, output_length) > MAX_FST_RESULT_WORD_LENGTH:
                raise _validation_error(
                    "replay_labels_exceed_bound",
                    "replayed labels exceed the result word bound",
                )
        return self


class RelationPathReplayResult(RelationPathReplayRequest):
    status: Literal["ACCEPTING_PAIR", "INVALID_PATH"]
    input_word: tuple[int, ...] = Field(max_length=MAX_FST_RESULT_WORD_LENGTH)
    output_word: tuple[int, ...] = Field(max_length=MAX_FST_RESULT_WORD_LENGTH)
    state_trace: tuple[int, ...] = Field(max_length=MAX_FST_WORD_LENGTH + 1)
    error: str | None = None

    @model_validator(mode="after")
    def bind_replay(self) -> Self:
        from jacobian.math.finite_state_transducers.operations import (
            replay_rational_path,
        )

        expected = replay_rational_path(
            self.transducer, self.initial_state, self.edge_path
        )
        actual = (
            self.status,
            self.input_word,
            self.output_word,
            self.state_trace,
            self.error,
        )
        if actual != expected:
            raise _validation_error(
                "replay_result_mismatch",
                "result must match the exact bound path replay",
            )
        return self


__all__ = [
    "ComposeRequest",
    "ComposeResult",
    "RelationPathReplayRequest",
    "RelationPathReplayResult",
    "SubseqRunRequest",
    "SubseqRunResult",
]
