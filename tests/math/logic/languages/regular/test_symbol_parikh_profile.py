"""Exact accepted-word symbol Parikh profiles."""

from itertools import product

import pytest

from jacobian.math.logic.languages.regular._symbol_parikh import (
    SymbolParikhProfileRequest,
    SymbolParikhProfileResult,
    symbol_parikh_profile,
)
from jacobian.math.logic.languages.regular.operations import (
    count_accepted_words,
    dfa_run,
)
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
    assert result.total_accepted_words == count_accepted_words(dfa, 3)
    assert (
        SymbolParikhProfileResult.model_validate_json(result.model_dump_json())
        == result
    )


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

    def fail(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("profile kernel must not replay accepted-word counting")

    # The pre-fix implementation imported this binding into the profile
    # module. Patching that binding makes the regression fail on the base if
    # the profile still delegates to the separate count operation.
    monkeypatch.setattr(profile, "count_accepted_words", fail, raising=False)
    result = profile.symbol_parikh_profile(
        profile.SymbolParikhProfileRequest(dfa=ending_in_one(), word_length=3)
    )
    assert result.total_accepted_words == 4


def test_wide_alphabet_uses_only_extension_layers_in_admission() -> None:
    alphabet_size = 32
    dfa = DFA(
        state_count=1,
        alphabet_size=alphabet_size,
        transitions=tuple(
            DFATransition(source=0, symbol=symbol, target=0)
            for symbol in range(alphabet_size)
        ),
        initial_state=0,
        accepting_states=(0,),
    )

    result = symbol_parikh_profile(SymbolParikhProfileRequest(dfa=dfa, word_length=3))

    assert len(result.cells) == 5_984
    assert result.total_accepted_words == 32**3


def test_one_symbol_profile_retains_its_axis_at_the_length_limit() -> None:
    dfa = DFA(
        state_count=1,
        alphabet_size=1,
        transitions=(DFATransition(source=0, symbol=0, target=0),),
        initial_state=0,
        accepting_states=(0,),
    )

    result = symbol_parikh_profile(
        SymbolParikhProfileRequest(dfa=dfa, word_length=1_000)
    )

    assert result.alphabet == (0,)
    assert result.cells[0].symbol_counts == (1_000,)
    assert result.cells[0].multiplicity == 1
