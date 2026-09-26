"""Typed values for deterministic coaccessibility witnesses."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.automata.transducers.values import (
    MAX_FST_RESULT_WORD_LENGTH,
    MAX_FST_STATES,
    MAX_FST_WORD_LENGTH,
    SubsequentialTransducer,
)


class CoaccessibleStatesRequest(StrictModel):
    """Find a successful continuation from every coaccessible state."""

    transducer: SubsequentialTransducer


class CoaccessibleStateWitness(StrictModel):
    """One shortest, lexicographically first successful continuation."""

    state: int = Field(ge=0, lt=MAX_FST_STATES)
    input_suffix: tuple[int, ...] = Field(max_length=MAX_FST_WORD_LENGTH)
    state_trace: tuple[int, ...] = Field(max_length=MAX_FST_STATES)
    transition_indices: tuple[int, ...] = Field(max_length=MAX_FST_STATES - 1)
    final_state: int = Field(ge=0, lt=MAX_FST_STATES)
    output_word: tuple[int, ...] = Field(max_length=MAX_FST_RESULT_WORD_LENGTH)

    @model_validator(mode="after")
    def require_well_shaped_witness(self) -> Self:
        if (
            not self.state_trace
            or self.state_trace[0] != self.state
            or self.state_trace[-1] != self.final_state
            or len(self.state_trace) != len(self.input_suffix) + 1
            or len(self.transition_indices) != len(self.input_suffix)
        ):
            raise PydanticCustomError(
                "finite_state_transducer.coaccessible_witness_shape",
                "coaccessibility witness path fields have incompatible lengths",
            )
        return self


class CoaccessibleStateWitnesses(CoaccessibleStatesRequest):
    """Source-bound shortest successful continuation for each such state."""

    witnesses: tuple[CoaccessibleStateWitness, ...] = Field(max_length=MAX_FST_STATES)

    @model_validator(mode="after")
    def require_canonical_state_order(self) -> Self:
        states = tuple(witness.state for witness in self.witnesses)
        if states != tuple(sorted(set(states))):
            raise PydanticCustomError(
                "finite_state_transducer.coaccessible_states_not_canonical",
                "coaccessible state witnesses must be state sorted",
            )
        source = self.transducer
        if any(state < 0 or state >= source.state_count for state in states):
            raise PydanticCustomError(
                "finite_state_transducer.coaccessible_state_out_of_range",
                "coaccessible state witness is outside the source",
            )
        transitions = source.transitions
        for witness in self.witnesses:
            if any(
                state < 0 or state >= source.state_count
                for state in witness.state_trace
            ):
                raise PydanticCustomError(
                    "finite_state_transducer.coaccessible_trace_state_out_of_range",
                    "witness trace state is outside the source",
                )
            if any(
                symbol < 0 or symbol >= source.input_alphabet_size
                for symbol in witness.input_suffix
            ):
                raise PydanticCustomError(
                    "finite_state_transducer.coaccessible_input_symbol_out_of_range",
                    "witness input symbol is outside the source alphabet",
                )
            if any(
                symbol < 0 or symbol >= source.output_alphabet_size
                for symbol in witness.output_word
            ):
                raise PydanticCustomError(
                    "finite_state_transducer.coaccessible_output_symbol_out_of_range",
                    "witness output symbol is outside the source alphabet",
                )
            if any(
                index < 0 or index >= len(transitions)
                for index in witness.transition_indices
            ):
                raise PydanticCustomError(
                    "finite_state_transducer.coaccessible_transition_index_out_of_range",
                    "witness transition index is outside the source transitions",
                )
        return self


__all__ = [
    "CoaccessibleStateWitness",
    "CoaccessibleStateWitnesses",
    "CoaccessibleStatesRequest",
]
