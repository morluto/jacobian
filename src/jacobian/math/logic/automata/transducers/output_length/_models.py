"""Wire contracts for bounded subsequential output-length queries."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.automata.transducers.values import (
    MAX_FST_EDGES,
    MAX_FST_STATES,
    MAX_FST_WORD_LENGTH,
    SubsequentialTransducer,
)

MAX_SUBSEQUENTIAL_OUTPUT_LENGTH = (MAX_FST_WORD_LENGTH + 1) * MAX_FST_WORD_LENGTH
MAX_SUBSEQUENTIAL_OUTPUT_LENGTH_WORK = MAX_FST_WORD_LENGTH + MAX_FST_EDGES


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"finite_state_transducer.{reason}", message)


class SubsequentialOutputLengthRequest(StrictModel):
    """A bounded input word whose output length is queried."""

    transducer: SubsequentialTransducer
    word: tuple[int, ...] = Field(max_length=MAX_FST_WORD_LENGTH)

    @model_validator(mode="after")
    def require_input_symbols_in_alphabet(self) -> Self:
        if any(
            not 0 <= symbol < self.transducer.input_alphabet_size
            for symbol in self.word
        ):
            raise _error(
                "word_symbol_out_of_range",
                "input word symbol is outside its alphabet",
            )
        return self


class SubsequentialOutputLengthResult(StrictModel):
    """Exact output length or exact undefined-run information.

    `output_length` is present exactly when the transducer function is defined
    on the supplied word. For the two undefined outcomes,
    `transition_output_length` reports the number of symbols emitted before
    the function becomes undefined; it never materializes those symbols.
    """

    transducer: SubsequentialTransducer
    word: tuple[int, ...] = Field(max_length=MAX_FST_WORD_LENGTH)
    status: Literal["OUTPUT", "UNDEFINED_TRANSITION", "NONFINAL_DOMAIN_STATE"]
    output_length: int | None = Field(
        default=None, ge=0, le=MAX_SUBSEQUENTIAL_OUTPUT_LENGTH
    )
    transition_output_length: int = Field(ge=0, le=MAX_SUBSEQUENTIAL_OUTPUT_LENGTH)
    final_state: int = Field(ge=0, lt=MAX_FST_STATES)
    undefined_position: int | None = None

    @model_validator(mode="after")
    def require_canonical_outcome_shape(self) -> Self:
        if not 0 <= self.final_state < self.transducer.state_count:
            raise _error(
                "length_final_state_out_of_range",
                "final state is outside the transducer",
            )
        if self.status == "OUTPUT":
            valid = self.output_length is not None and self.undefined_position is None
        elif self.status == "UNDEFINED_TRANSITION":
            valid = (
                self.output_length is None
                and self.undefined_position is not None
                and 0 <= self.undefined_position < len(self.word)
            )
        else:
            valid = self.output_length is None and self.undefined_position is None
        if not valid:
            raise _error(
                "length_outcome_shape",
                "status and output-length fields are incompatible",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: SubsequentialOutputLengthRequest,
        *,
        status: Literal["OUTPUT", "UNDEFINED_TRANSITION", "NONFINAL_DOMAIN_STATE"],
        output_length: int | None,
        transition_output_length: int,
        final_state: int,
        undefined_position: int | None,
    ) -> Self:
        return cls.model_construct(
            transducer=request.transducer,
            word=request.word,
            status=status,
            output_length=output_length,
            transition_output_length=transition_output_length,
            final_state=final_state,
            undefined_position=undefined_position,
        )


__all__ = [
    "MAX_SUBSEQUENTIAL_OUTPUT_LENGTH",
    "MAX_SUBSEQUENTIAL_OUTPUT_LENGTH_WORK",
    "SubsequentialOutputLengthRequest",
    "SubsequentialOutputLengthResult",
]
