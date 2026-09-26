"""Exact regular-language restriction for partial subsequential functions."""

from __future__ import annotations

import json
from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.logic.automata.transducers.domain_restriction import (
    SubsequentialDomainRestrictionRequest,
    restrict_subsequential_domain,
)
from jacobian.math.logic.automata.transducers.values import (
    FiniteAlphabet,
    SubsequentialTransducer,
)
from jacobian.math.logic.languages.regular.values import DFA, DFATransition


def _parity_machine() -> tuple[SubsequentialTransducer, DFA]:
    alphabet = FiniteAlphabet(symbols=("a",))
    transducer = SubsequentialTransducer(
        input_alphabet_size=1,
        output_alphabet_size=1,
        input_alphabet_id="letter",
        output_alphabet_id="letter",
        input_alphabet=alphabet,
        output_alphabet=alphabet,
        state_count=1,
        initial_state=0,
        transitions=({"source": 0, "input_symbol": 0, "target": 0, "output": (0,)},),
        final_outputs=({"state": 0, "output": ()},),
    )
    dfa = DFA(
        state_count=2,
        alphabet_size=1,
        alphabet_id="letter",
        alphabet=alphabet,
        initial_state=0,
        accepting_states=(0,),
        transitions=(
            DFATransition(source=0, symbol=0, target=1),
            DFATransition(source=1, symbol=0, target=0),
        ),
    )
    return transducer, dfa


def _run(
    transducer: SubsequentialTransducer, word: tuple[int, ...]
) -> tuple[int, ...] | None:
    transitions = {
        (item.source, item.input_symbol): item for item in transducer.transitions
    }
    finals = {item.state: item.output for item in transducer.final_outputs}
    state = transducer.initial_state
    output: list[int] = []
    for symbol in word:
        edge = transitions.get((state, symbol))
        if edge is None:
            return None
        output.extend(edge.output)
        state = edge.target
    final = finals.get(state)
    if final is None:
        return None
    output.extend(final)
    return tuple(output)


def _accepts(dfa: DFA, word: tuple[int, ...]) -> bool:
    transitions = {(edge.source, edge.symbol): edge.target for edge in dfa.transitions}
    state = dfa.initial_state
    for symbol in word:
        state = transitions[(state, symbol)]
    return state in dfa.accepting_states


def test_restricted_machine_matches_independent_word_oracle() -> None:
    source, dfa = _parity_machine()
    restricted = restrict_subsequential_domain(
        SubsequentialDomainRestrictionRequest(transducer=source, domain_dfa=dfa)
    )

    assert restricted.input_alphabet == source.input_alphabet
    assert restricted.input_alphabet_id == source.input_alphabet_id
    assert restricted.output_alphabet == source.output_alphabet
    assert restricted.output_alphabet_id == source.output_alphabet_id
    for length in range(7):
        for word in product(range(1), repeat=length):
            expected = _run(source, word) if _accepts(dfa, word) else None
            assert _run(restricted, word) == expected


def test_rejected_domain_returns_one_state_empty_function() -> None:
    source, dfa = _parity_machine()
    rejecting = dfa.model_copy(update={"accepting_states": ()})
    restricted = restrict_subsequential_domain(
        SubsequentialDomainRestrictionRequest(transducer=source, domain_dfa=rejecting)
    )
    assert restricted.state_count == 1
    assert restricted.initial_state == 0
    assert restricted.transitions == ()
    assert restricted.final_outputs == ()


def test_restriction_preserves_partial_domain_and_final_output() -> None:
    alphabet = FiniteAlphabet(symbols=("a", "b"))
    source = SubsequentialTransducer(
        input_alphabet_size=2,
        output_alphabet_size=2,
        input_alphabet_id="input",
        output_alphabet_id="output",
        input_alphabet=alphabet,
        output_alphabet=alphabet,
        state_count=2,
        initial_state=0,
        transitions=({"source": 0, "input_symbol": 0, "target": 1, "output": (1,)},),
        final_outputs=({"state": 1, "output": (0,)},),
    )
    all_words = DFA(
        state_count=1,
        alphabet_size=2,
        alphabet_id="input",
        alphabet=alphabet,
        initial_state=0,
        accepting_states=(0,),
        transitions=(
            DFATransition(source=0, symbol=0, target=0),
            DFATransition(source=0, symbol=1, target=0),
        ),
    )
    restricted = restrict_subsequential_domain(
        SubsequentialDomainRestrictionRequest(transducer=source, domain_dfa=all_words)
    )
    assert _run(restricted, (0,)) == (1, 0)
    assert _run(restricted, ()) is None
    assert _run(restricted, (1,)) is None


def test_mismatched_alphabet_context_is_rejected() -> None:
    source, dfa = _parity_machine()
    other_alphabet = FiniteAlphabet(symbols=("x",))
    wrong_parent = dfa.model_copy(
        update={"alphabet_id": "other", "alphabet": other_alphabet}
    )
    with pytest.raises(ValidationError, match="domain_alphabet_identity_mismatch"):
        SubsequentialDomainRestrictionRequest(
            transducer=source, domain_dfa=wrong_parent
        )


def test_product_larger_than_result_carrier_is_refused_exactly() -> None:
    alphabet = FiniteAlphabet(symbols=("a",))
    source = SubsequentialTransducer(
        input_alphabet_size=1,
        output_alphabet_size=1,
        input_alphabet_id="letter",
        output_alphabet_id="letter",
        input_alphabet=alphabet,
        output_alphabet=alphabet,
        state_count=9,
        initial_state=0,
        transitions=tuple(
            {"source": state, "input_symbol": 0, "target": (state + 1) % 9}
            for state in range(9)
        ),
        final_outputs=tuple({"state": state, "output": ()} for state in range(9)),
    )
    dfa = DFA(
        state_count=8,
        alphabet_size=1,
        alphabet_id="letter",
        alphabet=alphabet,
        initial_state=0,
        accepting_states=tuple(range(8)),
        transitions=tuple(
            DFATransition(source=state, symbol=0, target=(state + 1) % 8)
            for state in range(8)
        ),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        restrict_subsequential_domain(
            SubsequentialDomainRestrictionRequest(transducer=source, domain_dfa=dfa)
        )
    assert error.value.errors()[0]["type"] == (
        "finite_state_transducer.domain_restriction_state_bound_exceeded"
    )


def test_tool_is_discoverable_and_has_a_valid_example() -> None:
    operation_id = "transducer.subsequential.restrict_domain.compute"
    catalog = Catalog.open()
    tool = catalog.operation(operation_id)
    assert tool is not None
    assert len(tool.examples) == 1
    result = invoke_operation(operation_id, tool.examples[0].input, catalog)
    restricted = tool.result_type.model_validate_json(json.dumps(result.output))
    assert isinstance(restricted, SubsequentialTransducer)
    assert restricted.state_count == 2
    assert len(restricted.final_outputs) == 1
    assert restricted.final_outputs[0].state == 0
    assert restricted.final_outputs[0].output == ()
