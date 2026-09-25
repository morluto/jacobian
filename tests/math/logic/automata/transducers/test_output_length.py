"""Exact and scale regressions for subsequential output lengths."""

import pytest
from pydantic import ValidationError

from jacobian.math.logic.automata.transducers.operations import run_subsequential
from jacobian.math.logic.automata.transducers.output_length._models import (
    MAX_SUBSEQUENTIAL_OUTPUT_LENGTH,
    SubsequentialOutputLengthRequest,
)
from jacobian.math.logic.automata.transducers.output_length._tools import TOOLS
from jacobian.math.logic.automata.transducers.output_length.operations import (
    subsequential_output_length,
)
from jacobian.math.logic.automata.transducers.values import (
    SubseqFinalOutput,
    SubseqTransition,
    SubsequentialTransducer,
)


def _transducer(
    *,
    transitions: tuple[SubseqTransition, ...],
    final_outputs: tuple[SubseqFinalOutput, ...],
    state_count: int = 2,
) -> SubsequentialTransducer:
    return SubsequentialTransducer(
        input_alphabet_size=2,
        output_alphabet_size=2,
        state_count=state_count,
        initial_state=0,
        transitions=transitions,
        final_outputs=final_outputs,
    )


def test_length_agrees_with_materialized_run_including_final_output() -> None:
    transducer = _transducer(
        transitions=(
            SubseqTransition(source=0, input_symbol=0, target=1, output=(1, 0)),
        ),
        final_outputs=(SubseqFinalOutput(state=1, output=(1,)),),
    )
    request = SubsequentialOutputLengthRequest(transducer=transducer, word=(0,))

    result = subsequential_output_length(request)
    run = run_subsequential(transducer, (0,))

    assert result.status == "OUTPUT"
    assert result.output_length == len(run.output) == 3
    assert result.transition_output_length == 2
    assert result.final_state == run.final_state


def test_empty_output_and_empty_input_are_defined_with_length_zero() -> None:
    transducer = _transducer(
        transitions=(SubseqTransition(source=0, input_symbol=0, target=1, output=()),),
        final_outputs=(
            SubseqFinalOutput(state=0, output=()),
            SubseqFinalOutput(state=1, output=()),
        ),
    )

    empty = subsequential_output_length(
        SubsequentialOutputLengthRequest(transducer=transducer, word=())
    )
    consumed = subsequential_output_length(
        SubsequentialOutputLengthRequest(transducer=transducer, word=(0,))
    )

    assert (empty.status, empty.output_length) == ("OUTPUT", 0)
    assert (consumed.status, consumed.output_length) == ("OUTPUT", 0)


def test_undefined_and_nonfinal_runs_report_emitted_prefix_length() -> None:
    transducer = _transducer(
        transitions=(
            SubseqTransition(source=0, input_symbol=0, target=1, output=(0, 1)),
        ),
        final_outputs=(SubseqFinalOutput(state=0, output=()),),
    )

    undefined = subsequential_output_length(
        SubsequentialOutputLengthRequest(transducer=transducer, word=(0, 1))
    )
    nonfinal = subsequential_output_length(
        SubsequentialOutputLengthRequest(transducer=transducer, word=(0,))
    )

    assert (undefined.status, undefined.undefined_position) == (
        "UNDEFINED_TRANSITION",
        1,
    )
    assert undefined.output_length is None
    assert undefined.transition_output_length == 2
    assert (nonfinal.status, nonfinal.undefined_position) == (
        "NONFINAL_DOMAIN_STATE",
        None,
    )
    assert nonfinal.output_length is None
    assert nonfinal.transition_output_length == 2


def test_exact_large_length_does_not_materialize_run_output() -> None:
    maximum_transition_output = (0,) * 512
    transducer = SubsequentialTransducer(
        input_alphabet_size=1,
        output_alphabet_size=1,
        state_count=1,
        initial_state=0,
        transitions=(
            SubseqTransition(
                source=0,
                input_symbol=0,
                target=0,
                output=maximum_transition_output,
            ),
        ),
        final_outputs=(SubseqFinalOutput(state=0, output=maximum_transition_output),),
    )
    word = (0,) * 512

    result = subsequential_output_length(
        SubsequentialOutputLengthRequest(transducer=transducer, word=word)
    )

    assert result.status == "OUTPUT"
    assert result.output_length == MAX_SUBSEQUENTIAL_OUTPUT_LENGTH == 262656
    assert result.transition_output_length == 512 * 512


def test_wrong_alphabet_symbol_and_overlong_word_rejected() -> None:
    transducer = _transducer(transitions=(), final_outputs=())

    with pytest.raises(ValidationError):
        SubsequentialOutputLengthRequest(transducer=transducer, word=(2,))
    with pytest.raises(ValidationError):
        SubsequentialOutputLengthRequest(transducer=transducer, word=(0,) * 513)


def test_manifest_example_executes() -> None:
    operation_id = "transducer.subsequential.output_length.compute"
    tool = next(tool for tool in TOOLS if tool.operation_id == operation_id)
    request = tool.request_type.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert result.output_length == 3
