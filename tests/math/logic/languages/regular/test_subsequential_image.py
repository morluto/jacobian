"""Exact regular images under parented subsequential transducers."""

import itertools

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.automata.transducers.values import (
    FiniteAlphabet,
    SubseqFinalOutput,
    SubseqTransition,
    SubsequentialTransducer,
)
from jacobian.math.logic.languages.regular import (
    dfa_subsequential_image,
    nfa_membership,
)
from jacobian.math.logic.languages.regular._models import SubsequentialImageRequest
from jacobian.math.logic.languages.regular.values import DFA, NFA, DFATransition


def carriers() -> tuple[DFA, SubsequentialTransducer]:
    input_alphabet = FiniteAlphabet(symbols=("x", "y"))
    output_alphabet = FiniteAlphabet(symbols=("a", "b", "c"))
    source = DFA(
        state_count=2,
        alphabet_size=2,
        alphabet_id="inputs",
        alphabet=input_alphabet,
        transitions=(
            DFATransition(source=0, symbol=0, target=1),
            DFATransition(source=0, symbol=1, target=0),
            DFATransition(source=1, symbol=0, target=1),
            DFATransition(source=1, symbol=1, target=0),
        ),
        initial_state=0,
        accepting_states=(1,),
    )
    transducer = SubsequentialTransducer(
        input_alphabet_size=2,
        output_alphabet_size=3,
        input_alphabet_id="inputs",
        output_alphabet_id="outputs",
        input_alphabet=input_alphabet,
        output_alphabet=output_alphabet,
        state_count=3,
        initial_state=0,
        transitions=(
            SubseqTransition(source=0, input_symbol=0, target=1, output=(0,)),
            SubseqTransition(source=0, input_symbol=1, target=0, output=()),
            SubseqTransition(source=1, input_symbol=0, target=2, output=(1, 2)),
            SubseqTransition(source=2, input_symbol=1, target=0, output=(0,)),
        ),
        final_outputs=(
            SubseqFinalOutput(state=0, output=(2,)),
            SubseqFinalOutput(state=1, output=()),
        ),
    )
    return source, transducer


def direct_image_membership(
    source: DFA,
    transducer: SubsequentialTransducer,
    word: tuple[int, ...],
    output: tuple[int, ...],
) -> bool:
    """Independent direct semantics for a fixed input word and output word."""
    source_edges = {
        (edge.source, edge.symbol): edge.target for edge in source.transitions
    }
    transducer_edges = {
        (edge.source, edge.input_symbol): edge for edge in transducer.transitions
    }
    final_outputs = {item.state: item.output for item in transducer.final_outputs}
    source_state = source.initial_state
    transducer_state = transducer.initial_state
    emitted: list[int] = []
    for input_symbol in word:
        source_state = source_edges[(source_state, input_symbol)]
        edge = transducer_edges.get((transducer_state, input_symbol))
        if edge is None:
            return False
        transducer_state = edge.target
        emitted.extend(edge.output)
    final_output = final_outputs.get(transducer_state)
    if source_state not in source.accepting_states or final_output is None:
        return False
    emitted.extend(final_output)
    return tuple(emitted) == output


def test_image_matches_independent_exhaustive_finite_language_oracle() -> None:
    source, transducer = carriers()
    result = dfa_subsequential_image(source, transducer)
    assert result.alphabet == transducer.output_alphabet
    assert result.alphabet_id == transducer.output_alphabet_id
    assert tuple(edge.transition_id for edge in result.transitions) == tuple(
        range(len(result.transitions))
    )
    assert any(edge.symbol is None for edge in result.transitions)
    restored = NFA.model_validate_json(result.model_dump_json(), strict=True)
    assert restored == result
    output_words = [
        output
        for length in range(7)
        for output in itertools.product(
            range(transducer.output_alphabet_size), repeat=length
        )
    ]
    for output in output_words:
        expected = any(
            direct_image_membership(source, transducer, input_word, output)
            for input_length in range(6)
            for input_word in itertools.product(
                range(source.alphabet_size), repeat=input_length
            )
        )
        assert nfa_membership(restored, output) == expected


def test_image_requires_matching_alphabet_identity() -> None:
    source, transducer = carriers()
    mismatch = source.model_copy(update={"alphabet_id": "different"})
    with pytest.raises(ValidationError, match="image_alphabet_identity_mismatch"):
        SubsequentialImageRequest(dfa=mismatch, transducer=transducer)


def test_unemitted_output_symbol_has_no_accepting_path() -> None:
    alphabet = FiniteAlphabet(symbols=("x", "y"))
    source = DFA(
        state_count=1,
        alphabet_size=2,
        alphabet_id="in",
        alphabet=alphabet,
        transitions=(
            DFATransition(source=0, symbol=0, target=0),
            DFATransition(source=0, symbol=1, target=0),
        ),
        initial_state=0,
        accepting_states=(0,),
    )
    output_alphabet = FiniteAlphabet(symbols=("a", "b", "c"))
    # The transducer emits only output symbols 0 and 1, so no NFA edge can
    # accept a word containing symbol 2.
    transducer = SubsequentialTransducer(
        input_alphabet_size=2,
        output_alphabet_size=3,
        input_alphabet_id="in",
        output_alphabet_id="out",
        input_alphabet=alphabet,
        output_alphabet=output_alphabet,
        state_count=1,
        initial_state=0,
        transitions=(
            SubseqTransition(source=0, input_symbol=0, target=0, output=(0,)),
            SubseqTransition(source=0, input_symbol=1, target=0, output=(1,)),
        ),
        final_outputs=(SubseqFinalOutput(state=0, output=()),),
    )
    result = dfa_subsequential_image(source, transducer)
    assert nfa_membership(result, (0, 1, 0))
    assert not nfa_membership(result, (2,))


def test_image_revalidates_model_constructed_carriers() -> None:
    source, transducer = carriers()
    malformed_source = DFA.model_construct(
        **{**source.__dict__, "transitions": source.transitions[:-1]}
    )
    malformed_transducer = SubsequentialTransducer.model_construct(
        **{
            **transducer.__dict__,
            "transitions": (*transducer.transitions, transducer.transitions[0]),
        }
    )
    with pytest.raises(OperationDomainValidationError, match="canonical DFA"):
        dfa_subsequential_image(malformed_source, transducer)
    with pytest.raises(
        OperationDomainValidationError, match="canonical subsequential transducer"
    ):
        dfa_subsequential_image(source, malformed_transducer)
