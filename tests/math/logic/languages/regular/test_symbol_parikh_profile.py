"""Exact accepted-word symbol Parikh profiles."""

from itertools import product

from jacobian.math.logic.languages.regular._symbol_parikh import (
    SymbolParikhProfileRequest,
    symbol_parikh_profile,
)
from jacobian.math.logic.languages.regular.operations import dfa_run
from jacobian.math.logic.languages.regular.values import DFA, DFATransition


def ending_in_one() -> DFA:
    return DFA(
        state_count=2,
        alphabet_size=2,
        transitions=tuple(
            DFATransition(source=source, symbol=symbol, target=symbol)
            for source in range(2)
            for symbol in range(2)
        ),
        initial_state=0,
        accepting_states=(1,),
    )


def test_profile_matches_independent_word_enumeration() -> None:
    dfa = ending_in_one()
    result = symbol_parikh_profile(SymbolParikhProfileRequest(dfa=dfa, word_length=3))
    oracle: dict[tuple[int, int], int] = {}
    for word in product(range(2), repeat=3):
        if dfa_run(dfa, word)[0]:
            counts = (word.count(0), word.count(1))
            oracle[counts] = oracle.get(counts, 0) + 1
    assert result.alphabet == (0, 1)
    assert {cell.symbol_counts: cell.multiplicity for cell in result.cells} == oracle
    assert result.total_accepted_words == 4


def test_length_zero_retains_empty_count_vector() -> None:
    dfa = ending_in_one()
    result = symbol_parikh_profile(SymbolParikhProfileRequest(dfa=dfa, word_length=0))
    assert result.cells == ()
    accepting = dfa.model_copy(update={"accepting_states": (0,)})
    accepted = symbol_parikh_profile(
        SymbolParikhProfileRequest(dfa=accepting, word_length=0)
    )
    assert accepted.cells[0].symbol_counts == (0, 0)
    assert accepted.cells[0].multiplicity == 1


def test_empty_alphabet_has_only_the_empty_word() -> None:
    dfa = DFA(
        state_count=1,
        alphabet_size=0,
        transitions=(),
        initial_state=0,
        accepting_states=(0,),
    )
    result = symbol_parikh_profile(SymbolParikhProfileRequest(dfa=dfa, word_length=0))
    assert result.alphabet == ()
    assert result.cells[0].symbol_counts == ()
    assert result.total_accepted_words == 1
