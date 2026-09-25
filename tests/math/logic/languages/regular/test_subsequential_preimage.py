"""Exact regular preimages under parented subsequential transducers."""

import itertools

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
)
from jacobian.math.logic.automata.transducers.values import (
    FiniteAlphabet,
    SubseqFinalOutput,
    SubseqTransition,
    SubsequentialTransducer,
)
from jacobian.math.logic.finite_alphabet import FiniteAlphabet as CanonicalAlphabet
from jacobian.math.logic.languages.regular import FiniteAlphabet as RegularAlphabet
from jacobian.math.logic.languages.regular._models import (
    EquivalenceRequest,
    SubsequentialPreimageRequest,
)
from jacobian.math.logic.languages.regular.operations import (
    dfa_complement,
    dfa_equivalence,
    dfa_run,
    dfa_subsequential_preimage,
)
from jacobian.math.logic.languages.regular.values import DFA, DFATransition


def carriers() -> tuple[DFA, SubsequentialTransducer]:
    output_alphabet = FiniteAlphabet(symbols=("a", "b"))
    dfa = DFA(
        state_count=2,
        alphabet_size=2,
        alphabet_id="letters",
        alphabet=output_alphabet,
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
        output_alphabet_size=2,
        input_alphabet_id="inputs",
        output_alphabet_id="letters",
        input_alphabet=FiniteAlphabet(symbols=("x", "y")),
        output_alphabet=output_alphabet,
        state_count=3,
        initial_state=0,
        transitions=(
            SubseqTransition(source=0, input_symbol=0, target=1, output=(0,)),
            SubseqTransition(source=1, input_symbol=0, target=1, output=(1,)),
            SubseqTransition(source=0, input_symbol=1, target=2, output=()),
            SubseqTransition(source=2, input_symbol=0, target=2, output=()),
        ),
        final_outputs=(
            SubseqFinalOutput(state=0, output=()),
            SubseqFinalOutput(state=1, output=(0,)),
        ),
    )
    return dfa, transducer


def oracle(
    dfa: DFA, transducer: SubsequentialTransducer, word: tuple[int, ...]
) -> bool:
    edges = {(t.source, t.input_symbol): t for t in transducer.transitions}
    finals = {f.state: f.output for f in transducer.final_outputs}
    state = transducer.initial_state
    output: list[int] = []
    for symbol in word:
        edge = edges.get((state, symbol))
        if edge is None:
            return False
        state = edge.target
        output.extend(edge.output)
    if state not in finals:
        return False
    output.extend(finals[state])
    return dfa_run(dfa, tuple(output))[0]


def test_preimage_matches_independent_exhaustive_input_word_oracle() -> None:
    dfa, transducer = carriers()
    result = dfa_subsequential_preimage(dfa, transducer)
    assert CanonicalAlphabet is RegularAlphabet is FiniteAlphabet
    assert result.alphabet == transducer.input_alphabet
    assert result.alphabet_id == transducer.input_alphabet_id
    for length in range(6):
        for word in itertools.product(range(2), repeat=length):
            assert dfa_run(result, word)[0] == oracle(dfa, transducer, word)


def test_parent_round_trip_complement_and_equivalence() -> None:
    dfa, transducer = carriers()
    preimage = dfa_subsequential_preimage(dfa, transducer)
    restored = DFA.model_validate_json(preimage.model_dump_json(), strict=True)
    assert restored == preimage
    assert dfa_complement(restored).alphabet == transducer.input_alphabet
    assert dfa_complement(restored).alphabet_id == "inputs"
    assert dfa_equivalence(preimage, restored) == (None, None, None)


def test_cross_domain_preimage_requires_same_explicit_alphabet_parent() -> None:
    dfa, transducer = carriers()
    mismatch = dfa.model_copy(update={"alphabet": FiniteAlphabet(symbols=("a", "c"))})
    with pytest.raises(ValueError, match="contexts must be identical"):
        SubsequentialPreimageRequest(dfa=mismatch, transducer=transducer)
    unparented = dfa.model_copy(update={"alphabet": None})
    with pytest.raises(ValueError, match="explicit DFA"):
        SubsequentialPreimageRequest(dfa=unparented, transducer=transducer)
    with pytest.raises(OperationDomainValidationError, match="matching explicit"):
        dfa_subsequential_preimage(mismatch, transducer)

    different_identity = dfa.model_copy(update={"alphabet_id": "another-letters"})
    with pytest.raises(ValueError, match="identities must be identical"):
        SubsequentialPreimageRequest(dfa=different_identity, transducer=transducer)


def test_preimage_accepts_tiny_reachable_product_with_large_unreachable_carriers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alphabet = FiniteAlphabet(symbols=("a",))
    dfa = DFA(
        state_count=8,
        alphabet_size=1,
        alphabet_id="output",
        alphabet=alphabet,
        transitions=tuple(
            DFATransition(source=state, symbol=0, target=state) for state in range(8)
        ),
        initial_state=0,
        accepting_states=(0,),
    )
    transducer = SubsequentialTransducer(
        input_alphabet_size=1,
        output_alphabet_size=1,
        input_alphabet_id="input",
        output_alphabet_id="output",
        input_alphabet=FiniteAlphabet(symbols=("x",)),
        output_alphabet=alphabet,
        state_count=9,
        initial_state=0,
        transitions=tuple(
            SubseqTransition(source=state, input_symbol=0, target=state)
            for state in range(9)
        ),
        final_outputs=(SubseqFinalOutput(state=0, output=()),),
    )

    result = dfa_subsequential_preimage(dfa, transducer)
    assert result.state_count == 1
    assert result.transitions[0].target == 0


def test_preimage_revalidates_model_constructed_carriers() -> None:
    dfa, transducer = carriers()
    malformed_dfa = DFA.model_construct(
        **{**dfa.__dict__, "transitions": dfa.transitions[:-1]}
    )
    malformed_transducer = SubsequentialTransducer.model_construct(
        **{
            **transducer.__dict__,
            "transitions": (*transducer.transitions, transducer.transitions[0]),
        }
    )
    with pytest.raises(OperationDomainValidationError, match="canonical DFA"):
        dfa_subsequential_preimage(malformed_dfa, transducer)
    with pytest.raises(
        OperationDomainValidationError, match="canonical subsequential transducer"
    ):
        dfa_subsequential_preimage(dfa, malformed_transducer)


def test_equivalence_rejects_equal_sized_but_different_parents() -> None:
    dfa, _ = carriers()
    other = dfa.model_copy(
        update={"alphabet_id": "other", "transitions": dfa.transitions}
    )
    with pytest.raises(ValueError, match="matching alphabet sizes and parents"):
        EquivalenceRequest(left=dfa, right=other)
