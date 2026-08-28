from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.math.logic.languages.regular._models import (
    ComplementRequest,
    CountRequest,
    RunRequest,
)
from jacobian.math.logic.languages.regular._operations import (
    compute_complement,
    compute_count,
    compute_run,
)
from jacobian.math.logic.languages.regular.operations import (
    count_accepted_words,
    dfa_complement,
    dfa_run,
)
from jacobian.math.logic.languages.regular.values import DFA, DFATransition


def _error_type(exc_info: pytest.ExceptionInfo[ValidationError]) -> str:
    return str(exc_info.value.errors()[0]["type"])


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


def _dfa_full_alphabet_accepting() -> DFA:
    """One accepting state looping on all 32 symbols."""

    return DFA(
        state_count=1,
        alphabet_size=32,
        transitions=tuple(
            DFATransition(source=0, symbol=symbol, target=0) for symbol in range(32)
        ),
        initial_state=0,
        accepting_states=(0,),
    )


def test_run_accepts_word_ending_in_1() -> None:
    dfa = _dfa_ends_in_1()
    result = compute_run(RunRequest(dfa=dfa, word=(1, 0, 1)))
    assert result.accepted is True
    assert result.final_state == 1
    assert result.state_trace == (0, 1, 0, 1)


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
    assert result.count == "4"


def test_count_length_zero() -> None:
    dfa = _dfa_ends_in_1()
    result = compute_count(CountRequest(dfa=dfa, word_length=0))
    assert result.count == "0"  # initial state 0 is not accepting


def test_count_length_zero_accepting() -> None:
    dfa = _dfa_even_zeros()
    result = compute_count(CountRequest(dfa=dfa, word_length=0))
    assert result.count == "1"  # initial state 0 is accepting


def test_count_even_zeros_length_3() -> None:
    """Length 3 binary strings with even number of 0s: 4 (111, 100, 010, 001)."""

    dfa = _dfa_even_zeros()
    result = compute_count(CountRequest(dfa=dfa, word_length=3))
    assert result.count == "4"


def test_count_matches_brute_force() -> None:
    """Count via matrix powering matches brute-force enumeration."""

    dfa = _dfa_ends_in_1()
    for length in range(8):
        result = compute_count(CountRequest(dfa=dfa, word_length=length))
        brute = 0
        for word in product(range(2), repeat=length):
            run = compute_run(RunRequest(dfa=dfa, word=word))
            if run.accepted:
                brute += 1
        assert result.count == str(brute), f"Length {length}: {result.count} != {brute}"


def test_count_exact_matrix_powering_known_answer() -> None:
    """Binary strings of length 10 ending in 1: 2**9 = 512."""

    dfa = _dfa_ends_in_1()
    result = compute_count(CountRequest(dfa=dfa, word_length=10))
    assert result.count == "512"


def test_count_large_value_uses_canonical_string() -> None:
    """A 32-symbol one-state DFA accepts all 32**200 words of length 200."""

    dfa = _dfa_full_alphabet_accepting()
    result = compute_count(CountRequest(dfa=dfa, word_length=200))
    assert result.count == str(32**200)


def test_run_and_count_results_remain_structural() -> None:
    dfa = _dfa_ends_in_1()
    run = compute_run(RunRequest(dfa=dfa, word=(1, 0, 1)))
    run_payload = run.model_dump()
    run_payload["accepted"] = False
    forged_run = type(run).model_validate(run_payload)
    assert forged_run.accepted is False

    count = compute_count(CountRequest(dfa=dfa, word_length=3))
    count_payload = count.model_dump()
    count_payload["count"] = "5"
    forged_count = type(count).model_validate(count_payload)
    assert int(forged_count.count) == 5


def test_native_kernels_are_typed_and_consistent() -> None:
    """Public kernels compose directly on DFA values without wire adapters."""

    dfa = _dfa_ends_in_1()
    assert dfa_run(dfa, (1, 0, 1)) == (True, 1)
    assert count_accepted_words(dfa, 3) == 4
    complemented = dfa_complement(dfa)
    assert isinstance(complemented, DFA)
    assert complemented.accepting_states == (0,)
    assert dfa_run(complemented, (0,))[0] is True


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
    with pytest.raises(ValidationError) as exc_info:
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
    assert _error_type(exc_info) == "regular_language.dfa_not_deterministic"


def test_contract_rejects_invalid_accepting_states() -> None:
    with pytest.raises(ValidationError) as exc_info:
        DFA(
            state_count=2,
            alphabet_size=2,
            transitions=(
                DFATransition(source=0, symbol=0, target=0),
                DFATransition(source=0, symbol=1, target=0),
                DFATransition(source=1, symbol=0, target=0),
                DFATransition(source=1, symbol=1, target=0),
            ),
            initial_state=0,
            accepting_states=(5,),  # out of range
        )
    assert _error_type(exc_info) == "regular_language.accepting_state_out_of_range"


def test_contract_rejects_out_of_range_word_symbol() -> None:
    dfa = _dfa_ends_in_1()
    with pytest.raises(ValueError, match="word symbols"):
        compute_run(
            RunRequest(dfa=dfa, word=(5,))
        )  # symbol 5 is out of range for alphabet_size=2


def test_contract_rejects_non_total_dfa() -> None:
    """A DFA missing a transition for some (state, symbol) pair must be rejected."""

    with pytest.raises(ValidationError) as exc_info:
        DFA(
            state_count=2,
            alphabet_size=2,
            transitions=(
                DFATransition(source=0, symbol=0, target=0),
                DFATransition(source=0, symbol=1, target=1),
                DFATransition(source=1, symbol=0, target=0),
                # missing (1, symbol=1)
            ),
            initial_state=0,
            accepting_states=(1,),
        )
    assert _error_type(exc_info) == "regular_language.dfa_not_total"


def test_contract_rejects_review_missing_edge_example() -> None:
    """The reviewed one-state example must fail validation, not self-loop."""

    with pytest.raises(ValidationError) as exc_info:
        DFA(
            state_count=1,
            alphabet_size=1,
            transitions=(),
            initial_state=0,
            accepting_states=(0,),
        )
    assert _error_type(exc_info) == "regular_language.dfa_not_total"
