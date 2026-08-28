"""Typed wire contracts for exact bounded finite-state transducers."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.automata.transducers.values import (
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


class SubseqRunResult(SubseqRunRequest):
    """A bounded subsequential-run outcome.

    Deserialization checks only the canonical shape of a claimed outcome.  The
    owner-local verifier replays independently supplied claims; trusted kernel
    output is constructed through ``_from_kernel`` below.
    """

    status: Literal["OUTPUT", "UNDEFINED_TRANSITION", "NONFINAL_DOMAIN_STATE"]
    output: tuple[int, ...] = Field(max_length=MAX_FST_RESULT_WORD_LENGTH)
    final_state: int = Field(ge=0, lt=MAX_FST_STATES)
    undefined_position: int | None = None
    partial_output: tuple[int, ...] = Field(max_length=MAX_FST_RESULT_WORD_LENGTH)

    @model_validator(mode="after")
    def require_canonical_outcome_shape(self) -> Self:
        if not 0 <= self.final_state < self.transducer.state_count:
            raise _validation_error(
                "run_final_state_out_of_range", "final state is outside the transducer"
            )
        if any(
            not 0 <= symbol < self.transducer.output_alphabet_size
            for symbol in (*self.output, *self.partial_output)
        ):
            raise _validation_error(
                "run_output_symbol_out_of_range",
                "run output contains a symbol outside the output alphabet",
            )
        if self.status == "OUTPUT":
            valid = self.undefined_position is None and not self.partial_output
        elif self.status == "UNDEFINED_TRANSITION":
            valid = (
                not self.output
                and self.undefined_position is not None
                and 0 <= self.undefined_position < len(self.word)
            )
        else:
            valid = not self.output and self.undefined_position is None
        if not valid:
            raise _validation_error(
                "run_outcome_shape",
                "status and run outcome fields have an incompatible shape",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: SubseqRunRequest,
        *,
        status: Literal["OUTPUT", "UNDEFINED_TRANSITION", "NONFINAL_DOMAIN_STATE"],
        output: tuple[int, ...],
        final_state: int,
        undefined_position: int | None,
        partial_output: tuple[int, ...],
    ) -> Self:
        """Construct a run outcome emitted by the trusted owner-local kernel."""

        return cls(
            **request.model_dump(),
            status=status,
            output=output,
            final_state=final_state,
            undefined_position=undefined_position,
            partial_output=partial_output,
        )


class ComposeRequest(StrictModel):
    first: SubsequentialTransducer
    second: SubsequentialTransducer


class ComposeResult(ComposeRequest):
    transducer: SubsequentialTransducer

    @model_validator(mode="after")
    def require_composite_shape(self) -> Self:
        if (
            self.transducer.input_alphabet_size != self.first.input_alphabet_size
            or self.transducer.output_alphabet_size != self.second.output_alphabet_size
            or self.transducer.state_count
            > self.first.state_count * self.second.state_count
        ):
            raise _validation_error(
                "composition_result_shape",
                "composite transducer must retain composition alphabets and product-state bound",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: ComposeRequest, *, transducer: SubsequentialTransducer
    ) -> Self:
        """Construct a composition emitted by the trusted owner-local kernel."""

        return cls(**request.model_dump(), transducer=transducer)


class RelationPathReplayRequest(StrictModel):
    transducer: RationalTransducer
    initial_state: int = Field(ge=0, lt=MAX_FST_STATES)
    edge_path: tuple[int, ...] = Field(max_length=MAX_FST_WORD_LENGTH)


class RelationPathReplayResult(RelationPathReplayRequest):
    status: Literal["ACCEPTING_PAIR", "INVALID_PATH"]
    input_word: tuple[int, ...] = Field(max_length=MAX_FST_RESULT_WORD_LENGTH)
    output_word: tuple[int, ...] = Field(max_length=MAX_FST_RESULT_WORD_LENGTH)
    state_trace: tuple[int, ...] = Field(max_length=MAX_FST_WORD_LENGTH + 1)
    error: str | None = None

    @model_validator(mode="after")
    def require_canonical_replay_shape(self) -> Self:
        if (
            not self.state_trace
            or self.state_trace[0] != self.initial_state
            or len(self.state_trace) > len(self.edge_path) + 1
            or any(
                not 0 <= state < self.transducer.state_count
                for state in self.state_trace
            )
            or any(
                not 0 <= symbol < self.transducer.input_alphabet_size
                for symbol in self.input_word
            )
            or any(
                not 0 <= symbol < self.transducer.output_alphabet_size
                for symbol in self.output_word
            )
        ):
            raise _validation_error(
                "replay_result_shape",
                "path replay fields have an invalid canonical shape",
            )
        if self.status == "ACCEPTING_PAIR" and self.error is not None:
            raise _validation_error(
                "replay_accepting_error", "an accepting path cannot carry an error"
            )
        if self.status == "INVALID_PATH" and not self.error:
            raise _validation_error(
                "replay_invalid_missing_error",
                "an invalid path must explain its failure",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: RelationPathReplayRequest,
        *,
        status: Literal["ACCEPTING_PAIR", "INVALID_PATH"],
        input_word: tuple[int, ...],
        output_word: tuple[int, ...],
        state_trace: tuple[int, ...],
        error: str | None,
    ) -> Self:
        """Construct a path replay emitted by the trusted owner-local kernel."""

        return cls(
            **request.model_dump(),
            status=status,
            input_word=input_word,
            output_word=output_word,
            state_trace=state_trace,
            error=error,
        )


__all__ = [
    "ComposeRequest",
    "ComposeResult",
    "RelationPathReplayRequest",
    "RelationPathReplayResult",
    "SubseqRunRequest",
    "SubseqRunResult",
]
