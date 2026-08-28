"""Wire adapters for exact bounded finite-state transducers."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.automata.transducers._models import (
    ComposeRequest,
    ComposeResult,
    RelationPathReplayRequest,
    RelationPathReplayResult,
    SubseqRunRequest,
    SubseqRunResult,
)
from jacobian.math.logic.automata.transducers.operations import (
    compose_subsequential,
    replay_rational_path,
    run_subsequential,
)
from jacobian.math.logic.automata.transducers.values import (
    MAX_FST_RESULT_WORD_LENGTH,
    MAX_FST_STATES,
    MAX_FST_WORD_LENGTH,
)


def _reject(code: str, message: str, *location: str) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=f"finite_state_transducer.{code}",
        message=message,
    )


def _admit_run(request: SubseqRunRequest) -> None:
    if any(
        not 0 <= symbol < request.transducer.input_alphabet_size
        for symbol in request.word
    ):
        _reject(
            "word_symbol_out_of_range",
            "word symbol is outside the input alphabet",
            "word",
        )
    transition_bound = max(
        (len(transition.output) for transition in request.transducer.transitions),
        default=0,
    )
    final_bound = max(
        (len(final.output) for final in request.transducer.final_outputs), default=0
    )
    if len(request.word) * transition_bound + final_bound > MAX_FST_RESULT_WORD_LENGTH:
        _reject(
            "run_output_exceeds_bound",
            "subsequential output may exceed the result word bound",
            "word",
        )


def _admit_composition(request: ComposeRequest) -> None:
    if request.first.output_alphabet_size != request.second.input_alphabet_size:
        _reject(
            "composition_alphabet_mismatch",
            "first output alphabet must match second input alphabet",
            "first",
            "second",
        )
    if request.first.state_count * request.second.state_count > MAX_FST_STATES:
        _reject(
            "composition_state_bound_exceeded",
            f"composite product-state bound exceeds {MAX_FST_STATES}",
            "first",
            "second",
        )
    second_transition_bound = max(
        (len(transition.output) for transition in request.second.transitions), default=0
    )
    first_transition_bound = max(
        (len(transition.output) for transition in request.first.transitions), default=0
    )
    first_final_bound = max(
        (len(final.output) for final in request.first.final_outputs), default=0
    )
    second_final_bound = max(
        (len(final.output) for final in request.second.final_outputs), default=0
    )
    if first_transition_bound * second_transition_bound > MAX_FST_WORD_LENGTH:
        _reject(
            "composition_transition_output_exceeds_bound",
            "composite transition output may exceed the word bound",
            "first",
            "second",
        )
    if (
        first_final_bound * second_transition_bound + second_final_bound
        > MAX_FST_WORD_LENGTH
    ):
        _reject(
            "composition_final_output_exceeds_bound",
            "composite final output may exceed the word bound",
            "first",
            "second",
        )


def _admit_path(request: RelationPathReplayRequest) -> None:
    if request.initial_state not in request.transducer.initial_states:
        _reject(
            "initial_state_not_declared",
            "initial_state must select one declared initial state",
            "initial_state",
        )
    if all(0 <= index < len(request.transducer.edges) for index in request.edge_path):
        input_length = sum(
            len(request.transducer.edges[index].input_label)
            for index in request.edge_path
        )
        output_length = sum(
            len(request.transducer.edges[index].output_label)
            for index in request.edge_path
        )
        if max(input_length, output_length) > MAX_FST_RESULT_WORD_LENGTH:
            _reject(
                "replay_labels_exceed_bound",
                "replayed labels exceed the result word bound",
                "edge_path",
            )


def compute_run(request: SubseqRunRequest) -> SubseqRunResult:
    _admit_run(request)
    status, output, final_state, undefined_position, partial_output = run_subsequential(
        request.transducer, request.word
    )
    return SubseqRunResult._from_kernel(
        request,
        status=status,
        output=output,
        final_state=final_state,
        undefined_position=undefined_position,
        partial_output=partial_output,
    )


def compute_compose(request: ComposeRequest) -> ComposeResult:
    _admit_composition(request)
    return ComposeResult._from_kernel(
        request,
        transducer=compose_subsequential(request.first, request.second),
    )


def compute_relation_path_replay(
    request: RelationPathReplayRequest,
) -> RelationPathReplayResult:
    _admit_path(request)
    status, input_word, output_word, state_trace, error = replay_rational_path(
        request.transducer, request.initial_state, request.edge_path
    )
    return RelationPathReplayResult._from_kernel(
        request,
        status=status,
        input_word=input_word,
        output_word=output_word,
        state_trace=state_trace,
        error=error,
    )


__all__ = [
    "compute_compose",
    "compute_relation_path_replay",
    "compute_run",
]
