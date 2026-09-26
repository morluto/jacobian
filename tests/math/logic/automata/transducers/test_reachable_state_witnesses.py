from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.transducers import reachable_state_witnesses
from jacobian.math.logic.automata.transducers._models import (
    ReachableStatesRequest,
    ReachableStatesResult,
)
from jacobian.math.logic.automata.transducers._tools import TOOLS
from jacobian.math.logic.automata.transducers.values import (
    SubseqFinalOutput,
    SubseqTransition,
    SubsequentialTransducer,
)


def _direct_path(
    transducer: SubsequentialTransducer, word: tuple[int, ...]
) -> tuple[int, tuple[int, ...], tuple[int, ...]] | None:
    transitions = {
        (row.source, row.input_symbol): row for row in transducer.transitions
    }
    state = transducer.initial_state
    trace = [state]
    output: list[int] = []
    for symbol in word:
        transition = transitions.get((state, symbol))
        if transition is None:
            return None
        state = transition.target
        trace.append(state)
        output.extend(transition.output)
    return state, tuple(trace), tuple(output)


def test_shortest_witnesses_match_exhaustive_finite_path_oracle() -> None:
    transducer = SubsequentialTransducer(
        input_alphabet_size=2,
        output_alphabet_size=2,
        state_count=5,
        initial_state=0,
        transitions=(
            SubseqTransition(source=0, input_symbol=0, target=1, output=(1, 0)),
            SubseqTransition(source=0, input_symbol=1, target=2, output=()),
            SubseqTransition(source=1, input_symbol=0, target=0, output=(0,)),
            SubseqTransition(source=1, input_symbol=1, target=3, output=(0,)),
            SubseqTransition(source=2, input_symbol=0, target=3, output=(1,)),
            SubseqTransition(source=3, input_symbol=1, target=3, output=(1,)),
        ),
        final_outputs=(SubseqFinalOutput(state=3, output=(1, 0)),),
    )

    # Independently enumerate every path of length at most |Q|-1. A shortest
    # path never repeats a state, and tuple order supplies the tie-break oracle.
    shortest: dict[int, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = {}
    for length in range(transducer.state_count):
        for word in product(range(transducer.input_alphabet_size), repeat=length):
            path = _direct_path(transducer, word)
            if path is None:
                continue
            state, trace, output = path
            candidate = (word, trace, output)
            prior = shortest.get(state)
            if prior is None or (len(word), word) < (len(prior[0]), prior[0]):
                shortest[state] = candidate

    result = reachable_state_witnesses(transducer)
    assert tuple(row.state for row in result.witnesses) == tuple(sorted(shortest))
    for row in result.witnesses:
        expected_word, expected_trace, expected_output = shortest[row.state]
        assert (row.input_word, row.state_trace, row.output_word) == (
            expected_word,
            expected_trace,
            expected_output,
        )
    # A final output belongs to completed function evaluation, not a path
    # ending at a reachable state.
    witness_for_three = next(row for row in result.witnesses if row.state == 3)
    assert witness_for_three.input_word == (0, 1)
    assert witness_for_three.output_word == (1, 0, 0)


def test_catalog_contract_and_source_context() -> None:
    transducer = SubsequentialTransducer(
        input_alphabet_size=1,
        output_alphabet_size=1,
        state_count=2,
        initial_state=0,
        transitions=(SubseqTransition(source=0, input_symbol=0, target=1),),
        final_outputs=(),
    )
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "transducer.subsequential.reachable_states.compute"
    )
    result = tool.run(ReachableStatesRequest(transducer=transducer))
    assert result.transducer == transducer
    assert tuple(row.state for row in result.witnesses) == (0, 1)
    assert result.witnesses[1].input_word == (0,)
    assert result.witnesses[1].state_trace == (0, 1)
    example_request = tool.request_type.model_validate(tool.examples[0].input)
    example_result = tool.run(example_request)
    assert tuple(row.state for row in example_result.witnesses) == (0, 1, 2)
    decoded_result = tool.result_type.model_validate(
        example_result.model_dump(mode="json")
    )
    assert decoded_result == example_result


def test_shortest_witness_output_growth_is_preflighted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.logic.automata.transducers import operations

    transducer = SubsequentialTransducer(
        input_alphabet_size=1,
        output_alphabet_size=1,
        state_count=3,
        initial_state=0,
        transitions=(
            SubseqTransition(source=0, input_symbol=0, target=1, output=(0,) * 512),
            SubseqTransition(source=1, input_symbol=0, target=2, output=(0,) * 512),
        ),
        final_outputs=(),
    )
    monkeypatch.setattr(operations, "MAX_FST_REACHABLE_WITNESS_OUTPUT_CELLS", 1)
    with pytest.raises(OperationResourceAdmissionError) as excinfo:
        reachable_state_witnesses(transducer)
    assert (
        excinfo.value.errors()[0]["type"]
        == "finite_state_transducer.reachable_result_bytes_exceeded"
    )


def test_empty_witness_trace_is_rejected_as_validation_error() -> None:
    payload = {
        "transducer": {
            "input_alphabet_size": 1,
            "output_alphabet_size": 1,
            "state_count": 1,
            "initial_state": 0,
            "transitions": [],
            "final_outputs": [],
        },
        "witnesses": [
            {"state": 0, "input_word": [], "output_word": [], "state_trace": []}
        ],
    }
    with pytest.raises(ValidationError):
        ReachableStatesResult.model_validate(payload)
