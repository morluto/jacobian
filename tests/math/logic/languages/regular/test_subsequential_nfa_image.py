"""Exact epsilon-NFA images under parented subsequential transducers."""

import itertools

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.transducers.values import (
    FiniteAlphabet,
    SubseqFinalOutput,
    SubseqTransition,
    SubsequentialTransducer,
)
from jacobian.math.logic.languages.regular import (
    nfa_membership,
    nfa_subsequential_image,
)
from jacobian.math.logic.languages.regular._models import SubsequentialNFAImageRequest
from jacobian.math.logic.languages.regular.values import NFA, NFATransition


def finite_nfa() -> tuple[NFA, FiniteAlphabet]:
    alphabet = FiniteAlphabet(symbols=("x", "y"))
    return (
        NFA(
            state_count=4,
            alphabet_size=2,
            alphabet_id="input",
            alphabet=alphabet,
            transitions=(
                NFATransition(transition_id=0, source=0, symbol=None, target=1),
                NFATransition(transition_id=1, source=1, symbol=0, target=2),
                NFATransition(transition_id=2, source=1, symbol=1, target=3),
            ),
            initial_state=0,
            accepting_states=(1, 2, 3),
        ),
        alphabet,
    )


def sample_transducer(input_alphabet: FiniteAlphabet) -> SubsequentialTransducer:
    output_alphabet = FiniteAlphabet(symbols=("a", "b"))
    return SubsequentialTransducer(
        input_alphabet_size=2,
        output_alphabet_size=2,
        input_alphabet_id="input",
        output_alphabet_id="output",
        input_alphabet=input_alphabet,
        output_alphabet=output_alphabet,
        state_count=2,
        initial_state=0,
        transitions=(
            SubseqTransition(source=0, input_symbol=0, target=1, output=(0, 1)),
            SubseqTransition(source=0, input_symbol=1, target=0, output=()),
        ),
        final_outputs=(
            SubseqFinalOutput(state=0, output=(1,)),
            SubseqFinalOutput(state=1, output=()),
        ),
    )


def direct_output(
    transducer: SubsequentialTransducer, word: tuple[int, ...]
) -> tuple[int, ...] | None:
    edges = {(edge.source, edge.input_symbol): edge for edge in transducer.transitions}
    finals = {final.state: final.output for final in transducer.final_outputs}
    state = transducer.initial_state
    output: list[int] = []
    for symbol in word:
        edge = edges.get((state, symbol))
        if edge is None:
            return None
        state = edge.target
        output.extend(edge.output)
    final = finals.get(state)
    if final is None:
        return None
    output.extend(final)
    return tuple(output)


def words(alphabet_size: int, max_length: int) -> list[tuple[int, ...]]:
    return [
        word
        for length in range(max_length + 1)
        for word in itertools.product(range(alphabet_size), repeat=length)
    ]


def test_nfa_image_matches_exhaustive_finite_word_oracle() -> None:
    source, input_alphabet = finite_nfa()
    transducer = sample_transducer(input_alphabet)
    image = nfa_subsequential_image(source, transducer)
    assert image.alphabet == transducer.output_alphabet
    assert image.alphabet_id == transducer.output_alphabet_id
    restored = NFA.model_validate_json(image.model_dump_json(), strict=True)
    assert restored == image

    source_words = [word for word in words(2, 2) if nfa_membership(source, word)]
    expected = {
        output
        for input_word in source_words
        if (output := direct_output(transducer, input_word)) is not None
    }
    for output in words(2, 3):
        assert nfa_membership(restored, output) == (output in expected)
    assert expected == {(1,), (0, 1)}


def test_identity_image_preserves_nfa_language_and_empty_language_stays_empty() -> None:
    source, alphabet = finite_nfa()
    identity = SubsequentialTransducer(
        input_alphabet_size=2,
        output_alphabet_size=2,
        input_alphabet_id="input",
        output_alphabet_id="input",
        input_alphabet=alphabet,
        output_alphabet=alphabet,
        state_count=1,
        initial_state=0,
        transitions=(
            SubseqTransition(source=0, input_symbol=0, target=0, output=(0,)),
            SubseqTransition(source=0, input_symbol=1, target=0, output=(1,)),
        ),
        final_outputs=(SubseqFinalOutput(state=0, output=()),),
    )
    image = nfa_subsequential_image(source, identity)
    for word in words(2, 3):
        assert nfa_membership(image, word) == nfa_membership(source, word)

    empty_source = source.model_copy(update={"accepting_states": ()})
    empty_image = nfa_subsequential_image(empty_source, identity)
    assert empty_image.accepting_states == ()
    for word in words(2, 3):
        assert not nfa_membership(empty_image, word)


def test_nfa_image_requires_exact_alphabet_context() -> None:
    source, alphabet = finite_nfa()
    transducer = sample_transducer(alphabet)
    wrong_parent = source.model_copy(update={"alphabet_id": "other"})
    with pytest.raises(ValidationError, match="nfa_image_alphabet_identity_mismatch"):
        SubsequentialNFAImageRequest(nfa=wrong_parent, transducer=transducer)


def test_nfa_image_rejects_oversized_product_before_expansion(monkeypatch) -> None:
    import jacobian.math.logic.languages.regular.operations as operations

    alphabet = FiniteAlphabet(symbols=("x",))
    source = NFA(
        state_count=100_000,
        alphabet_size=1,
        alphabet_id="input",
        alphabet=alphabet,
        transitions=(),
        initial_state=0,
        accepting_states=(),
    )
    transducer = SubsequentialTransducer(
        input_alphabet_size=1,
        output_alphabet_size=1,
        input_alphabet_id="input",
        output_alphabet_id="output",
        input_alphabet=alphabet,
        output_alphabet=FiniteAlphabet(symbols=("a",)),
        state_count=2,
        initial_state=0,
        transitions=(),
        final_outputs=(),
    )

    def expansion_must_not_start(*args, **kwargs):
        pytest.fail("product construction ran before resource admission")

    monkeypatch.setattr(
        operations, "_build_nfa_subsequential_image", expansion_must_not_start
    )
    with pytest.raises(OperationResourceAdmissionError, match="product or output-NFA"):
        nfa_subsequential_image(source, transducer)
