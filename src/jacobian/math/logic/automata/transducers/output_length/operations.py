"""Exact kernels for subsequential output-length queries."""

from __future__ import annotations

from typing import Literal

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.transducers.operations import (
    _admit_transducer,
    _admit_word,
)
from jacobian.math.logic.automata.transducers.output_length._models import (
    MAX_SUBSEQUENTIAL_OUTPUT_LENGTH_WORK,
    SubsequentialOutputLengthResult,
)
from jacobian.math.logic.automata.transducers.values import MAX_FST_WORD_LENGTH


def subsequential_output_length(
    transducer: object,
    word: object,
) -> SubsequentialOutputLengthResult:
    """Return exact emitted-word length without constructing the word itself."""

    transducer = _admit_transducer(transducer)
    word = _admit_word(word, field="word")
    if len(word) > MAX_FST_WORD_LENGTH:
        raise OperationResourceAdmissionError(
            location=("word",),
            code="finite_state_transducer.output_length_word_bound",
            message="the input word exceeds the admitted length bound",
        )
    if any(not 0 <= symbol < transducer.input_alphabet_size for symbol in word):
        raise OperationDomainValidationError(
            location=("word",),
            code="finite_state_transducer.word_symbol_out_of_range",
            message="input word symbol is outside its alphabet",
        )
    work = len(transducer.transitions) + len(word)
    if work > MAX_SUBSEQUENTIAL_OUTPUT_LENGTH_WORK:
        raise OperationResourceAdmissionError(
            location=("transducer", "word"),
            code="finite_state_transducer.output_length_work_exceeded",
            message="transition indexing and input scan exceed the admitted work bound",
        )

    transition_map = {
        (row.source, row.input_symbol): (row.target, len(row.output))
        for row in transducer.transitions
    }
    final_outputs = {row.state: len(row.output) for row in transducer.final_outputs}
    state = transducer.initial_state
    emitted_length = 0
    for position, symbol in enumerate(word):
        step = transition_map.get((state, symbol))
        if step is None:
            return SubsequentialOutputLengthResult._from_kernel(
                transducer,
                word,
                status="UNDEFINED_TRANSITION",
                output_length=None,
                transition_output_length=emitted_length,
                final_state=state,
                undefined_position=position,
            )
        state, step_output_length = step
        emitted_length += step_output_length

    final_output_length = final_outputs.get(state)
    if final_output_length is None:
        status: Literal["OUTPUT", "UNDEFINED_TRANSITION", "NONFINAL_DOMAIN_STATE"] = (
            "NONFINAL_DOMAIN_STATE"
        )
        return SubsequentialOutputLengthResult._from_kernel(
            transducer,
            word,
            status=status,
            output_length=None,
            transition_output_length=emitted_length,
            final_state=state,
            undefined_position=None,
        )

    return SubsequentialOutputLengthResult._from_kernel(
        transducer,
        word,
        status="OUTPUT",
        output_length=emitted_length + final_output_length,
        transition_output_length=emitted_length,
        final_state=state,
        undefined_position=None,
    )


__all__ = ["subsequential_output_length"]
