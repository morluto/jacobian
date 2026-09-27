"""Exact shortest suffix witnesses for subsequential transducers."""

from __future__ import annotations

import pkgutil
from itertools import product

import pytest

import jacobian.math.logic.automata.transducers as transducer_package
from jacobian.math.logic.automata.transducers.coaccessibility_witnesses._models import (
    CoaccessibleStatesRequest,
)
from jacobian.math.logic.automata.transducers.coaccessibility_witnesses._tools import (
    TOOLS,
)
from jacobian.math.logic.automata.transducers.coaccessibility_witnesses.operations import (
    coaccessible_state_witnesses,
)
from jacobian.math.logic.automata.transducers.values import (
    MAX_FST_RESULT_WORD_LENGTH,
    SubseqFinalOutput,
    SubseqTransition,
    SubsequentialTransducer,
)


def _machine() -> SubsequentialTransducer:
    return SubsequentialTransducer(
        input_alphabet_size=2,
        output_alphabet_size=3,
        state_count=5,
        initial_state=0,
        transitions=(
            SubseqTransition(source=0, input_symbol=1, target=2, output=(1,)),
            SubseqTransition(source=0, input_symbol=0, target=1, output=()),
            SubseqTransition(source=2, input_symbol=0, target=1, output=(2,)),
            SubseqTransition(source=3, input_symbol=1, target=2, output=(0,)),
            SubseqTransition(source=4, input_symbol=0, target=4, output=(1,)),
        ),
        final_outputs=(
            SubseqFinalOutput(state=1, output=(0,)),
            SubseqFinalOutput(state=2, output=(1, 2)),
        ),
    )


def _direct_shortest(
    transducer: SubsequentialTransducer,
    start: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    transitions = {
        (edge.source, edge.input_symbol): edge for edge in transducer.transitions
    }
    finals = {row.state: row.output for row in transducer.final_outputs}
    for length in range(transducer.state_count):
        for suffix in product(range(transducer.input_alphabet_size), repeat=length):
            state = start
            trace = [state]
            outputs: list[int] = []
            for symbol in suffix:
                edge = transitions.get((state, symbol))
                if edge is None:
                    break
                state = edge.target
                trace.append(state)
                outputs.extend(edge.output)
            else:
                if state in finals:
                    outputs.extend(finals[state])
                    return tuple(suffix), tuple(trace), tuple(outputs)
    return None


def test_witnesses_match_independent_shortest_word_enumeration() -> None:
    transducer = _machine()
    result = coaccessible_state_witnesses(transducer)

    expected = {
        state: _direct_shortest(transducer, state)
        for state in range(transducer.state_count)
    }
    expected = {state: witness for state, witness in expected.items() if witness}
    assert tuple(row.state for row in result.witnesses) == tuple(expected)
    for row in result.witnesses:
        suffix, trace, output = expected[row.state]
        assert row.input_suffix == suffix
        assert row.state_trace == trace
        assert row.output_word == output
        assert row.transition_indices == tuple(
            transducer.transitions.index(
                next(
                    edge
                    for edge in transducer.transitions
                    if edge.source == trace[index]
                    and edge.input_symbol == suffix[index]
                )
            )
            for index in range(len(suffix))
        )


def test_shortest_then_lexical_choice_and_empty_successful_suffix() -> None:
    transducer = _machine()
    result = coaccessible_state_witnesses(transducer)
    by_state = {row.state: row for row in result.witnesses}

    # Symbols 0 and 1 both reach final states in one step; symbol 0 wins.
    assert by_state[0].input_suffix == (0,)
    assert by_state[0].output_word == (0,)
    assert by_state[1].input_suffix == ()
    assert by_state[1].output_word == (0,)
    assert 4 not in by_state


def test_round_trip_and_catalog_path() -> None:
    transducer = _machine()
    native = coaccessible_state_witnesses(transducer)
    restored = type(native).model_validate_json(native.model_dump_json())
    assert restored == native

    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "transducer.subsequential.coaccessible_states.compute"
    )
    assert tool.run(CoaccessibleStatesRequest(transducer=transducer)) == native


def test_owner_local_manifest_is_visible_to_catalog_discovery() -> None:
    manifests = tuple(
        module.name
        for module in pkgutil.walk_packages(
            transducer_package.__path__, f"{transducer_package.__name__}."
        )
        if module.name.endswith("._tools")
    )
    assert (
        "jacobian.math.logic.automata.transducers.coaccessibility_witnesses._tools"
        in manifests
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("state_trace", (-1, 1)),
        ("input_suffix", (-1,)),
        ("output_word", (-1,)),
        ("transition_indices", (-1,)),
    ],
)
def test_deserialization_rejects_negative_source_indices(
    field: str, replacement: tuple[int, ...]
) -> None:
    result = coaccessible_state_witnesses(_machine()).model_dump()
    result["witnesses"][0][field] = replacement

    with pytest.raises(ValueError):
        type(coaccessible_state_witnesses(_machine())).model_validate(result)


def test_longer_than_generic_result_word_bound_witness_is_accepted() -> None:
    transducer = SubsequentialTransducer(
        input_alphabet_size=1,
        output_alphabet_size=1,
        state_count=10,
        initial_state=0,
        transitions=tuple(
            SubseqTransition(
                source=state,
                input_symbol=0,
                target=state + 1,
                output=(0,) * 512,
            )
            for state in range(9)
        ),
        final_outputs=(SubseqFinalOutput(state=9, output=(0,) * 512),),
    )
    result = coaccessible_state_witnesses(transducer)
    assert len(result.witnesses[0].output_word) == 5120


def test_maximum_admitted_witness_output_length_is_accepted() -> None:
    transducer = SubsequentialTransducer(
        input_alphabet_size=1,
        output_alphabet_size=1,
        state_count=8,
        initial_state=0,
        transitions=tuple(
            SubseqTransition(
                source=state,
                input_symbol=0,
                target=state + 1,
                output=(0,) * 512,
            )
            for state in range(7)
        ),
        final_outputs=(SubseqFinalOutput(state=7, output=(0,) * 512),),
    )
    result = coaccessible_state_witnesses(transducer)
    assert len(result.witnesses[0].output_word) == MAX_FST_RESULT_WORD_LENGTH
