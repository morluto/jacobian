"""Exact accepted-word symbol Parikh profiles."""

from itertools import product

import pytest

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


def test_profile_result_rejects_noncanonical_claimed_cells() -> None:
    from jacobian.math.logic.languages.regular._symbol_parikh import (
        SymbolParikhCell,
        SymbolParikhProfileResult,
    )

    with pytest.raises(ValueError, match="nonnegative and sum"):
        SymbolParikhProfileResult(
            dfa=ending_in_one(),
            alphabet=(0, 1),
            word_length=3,
            cells=(SymbolParikhCell(symbol_counts=(4, -1), multiplicity=1),),
            total_accepted_words=1,
        )


def test_profile_does_not_replay_count_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.logic.languages.regular._symbol_parikh as profile
    import jacobian.math.logic.languages.regular.operations as regular_operations

    def fail(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("profile kernel must not replay accepted-word counting")

    monkeypatch.setattr(regular_operations, "count_accepted_words", fail)
    result = profile.symbol_parikh_profile(
        profile.SymbolParikhProfileRequest(dfa=ending_in_one(), word_length=3)
    )
    assert result.total_accepted_words == 4
