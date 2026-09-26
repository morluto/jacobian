"""Exact kernels for subsequential output-length queries."""

from __future__ import annotations

from typing import Literal

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.transducers.output_length._models import (
    MAX_SUBSEQUENTIAL_OUTPUT_LENGTH_WORK,
    SubsequentialOutputLengthRequest,
    SubsequentialOutputLengthResult,
)
from jacobian.math.logic.automata.transducers.values import SubsequentialTransducer


def subsequential_output_length(
    request: SubsequentialOutputLengthRequest | SubsequentialTransducer,
    word: tuple[int, ...] | None = None,
) -> SubsequentialOutputLengthResult:
    """Return exact emitted-word length without constructing the word itself."""

    if isinstance(request, SubsequentialTransducer):
        if word is None:
            raise OperationDomainValidationError(
                location=("word",),
                code="finite_state_transducer.output_length_word_required",
                message="a word is required when passing a transducer directly",
            )
        request = SubsequentialOutputLengthRequest(transducer=request, word=word)

    if not isinstance(request, SubsequentialOutputLengthRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="finite_state_transducer.output_length_request_type",
            message="request must be a SubsequentialOutputLengthRequest value",
        )
    try:
        request = SubsequentialOutputLengthRequest.model_validate(
            request.model_dump(), strict=True
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="finite_state_transducer.output_length_request_shape",
            message="request must satisfy its complete canonical shape",
        ) from exc
    # The request dump round-trip above revalidates the nested transducer value
    # once, including values created with model_construct().
    transducer = request.transducer
    word = request.word
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
                request,
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
            request,
            status=status,
            output_length=None,
            transition_output_length=emitted_length,
            final_state=state,
            undefined_position=None,
        )

    return SubsequentialOutputLengthResult._from_kernel(
        request,
        status="OUTPUT",
        output_length=emitted_length + final_output_length,
        transition_output_length=emitted_length,
        final_state=state,
        undefined_position=None,
    )


__all__ = ["subsequential_output_length"]
