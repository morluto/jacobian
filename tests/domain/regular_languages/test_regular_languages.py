from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.regular_languages import (
    DFA,
    ComplementRequest,
    CountRequest,
    DFATransition,
    RunRequest,
)
from jacobian.domains.regular_languages.operations import (
    compute_complement,
    compute_count,
    compute_run,
)


def _dfa_ends_in_1() -> DFA:
    return DFA(
        state_count=2,
        alphabet_size=2,
        transitions=(
            DFATransition(source=0, symbol=0, target=0),
            DFATransition(source=0, symbol=1, target=1),
            DFATransition(source=1, symbol=0, target=0),
            DFATransition(source=1, symbol=1, target=1),
        ),
        initial_state=0,
        accepting_states=(1,),
    )


def _dfa_even_zeros() -> DFA:
    """Accepts binary strings with an even number of 0s."""
    return DFA(
        state_count=2,
        alphabet_size=2,
        transitions=(
            DFATransition(source=0, symbol=0, target=1),
            DFATransition(source=0, symbol=1, target=0),
            DFATransition(source=1, symbol=0, target=0),
            DFATransition(source=1, symbol=1, target=1),
        ),
        initial_state=0,
        accepting_states=(0,),
    )


def test_run_accepts_word_ending_in_1() -> None:
    dfa = _dfa_ends_in_1()
    result = compute_run(RunRequest(dfa=dfa, word=(1, 0, 1)))
    assert result.accepted is True
    assert result.final_state == 1


def test_run_rejects_word_ending_in_0() -> None:
    dfa = _dfa_ends_in_1()
    result = compute_run(RunRequest(dfa=dfa, word=(1, 0, 0)))
    assert result.accepted is False
    assert result.final_state == 0


def test_run_accepts_empty_word_if_initial_is_accepting() -> None:
    dfa = _dfa_even_zeros()
    result = compute_run(RunRequest(dfa=dfa, word=()))
    assert result.accepted is True
    assert result.final_state == 0


def test_run_rejects_empty_word_if_initial_not_accepting() -> None:
    dfa = _dfa_ends_in_1()
    result = compute_run(RunRequest(dfa=dfa, word=()))
    assert result.accepted is False
    assert result.final_state == 0


def test_count_binary_strings_ending_in_1() -> None:
    """Length 3 binary strings ending in 1: 4 (001, 011, 101, 111)."""
    dfa = _dfa_ends_in_1()
    result = compute_count(CountRequest(dfa=dfa, word_length=3))
    assert result.count == 4


def test_count_length_zero() -> None:
    dfa = _dfa_ends_in_1()
    result = compute_count(CountRequest(dfa=dfa, word_length=0))
    assert result.count == 0  # initial state 0 is not accepting


def test_count_length_zero_accepting() -> None:
    dfa = _dfa_even_zeros()
    result = compute_count(CountRequest(dfa=dfa, word_length=0))
    assert result.count == 1  # initial state 0 is accepting


def test_count_even_zeros_length_3() -> None:
    """Length 3 binary strings with even number of 0s: 4 (111, 100, 010, 001)."""
    dfa = _dfa_even_zeros()
    result = compute_count(CountRequest(dfa=dfa, word_length=3))
    assert result.count == 4


def test_count_matches_brute_force() -> None:
    """Count via matrix powering matches brute-force enumeration."""
    from itertools import product

    dfa = _dfa_ends_in_1()
    for length in range(8):
        result = compute_count(CountRequest(dfa=dfa, word_length=length))
        brute = 0
        for word in product(range(2), repeat=length):
            res = compute_run(RunRequest(dfa=dfa, word=tuple(word)))
            if res.accepted:
                brute += 1
        assert result.count == brute, f"Length {length}: {result.count} != {brute}"


def test_complement_flips_acceptance() -> None:
    dfa = _dfa_ends_in_1()
    result = compute_complement(ComplementRequest(dfa=dfa))
    assert result.dfa.accepting_states == (0,)
    # Word ending in 0 should now be accepted
    run = compute_run(RunRequest(dfa=result.dfa, word=(0,)))
    assert run.accepted is True


def test_complement_double_complement_is_identity() -> None:
    dfa = _dfa_ends_in_1()
    result1 = compute_complement(ComplementRequest(dfa=dfa))
    result2 = compute_complement(ComplementRequest(dfa=result1.dfa))
    assert result2.dfa.accepting_states == dfa.accepting_states


def test_contract_rejects_duplicate_transitions() -> None:
    with pytest.raises(ValidationError, match="deterministic"):
        DFA(
            state_count=2,
            alphabet_size=2,
            transitions=(
                DFATransition(source=0, symbol=0, target=0),
                DFATransition(source=0, symbol=0, target=1),  # duplicate
            ),
            initial_state=0,
            accepting_states=(1,),
        )


def test_contract_rejects_invalid_accepting_states() -> None:
    with pytest.raises(ValidationError):
        DFA(
            state_count=2,
            alphabet_size=2,
            transitions=(),
            initial_state=0,
            accepting_states=(5,),  # out of range
        )


def test_contract_rejects_out_of_range_word_symbol() -> None:
    dfa = _dfa_ends_in_1()
    with pytest.raises(ValidationError, match="word symbols"):
        RunRequest(dfa=dfa, word=(5,))  # symbol 5 is out of range for alphabet_size=2
