from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.languages.regular._models import (
    ComplementRequest,
    CountRequest,
    RunRequest,
)
from jacobian.math.logic.languages.regular._tools import (
    compute_complement,
    compute_count,
    compute_run,
)
from jacobian.math.logic.languages.regular.operations import (
    count_accepted_words,
    dfa_complement,
    dfa_run,
)
from jacobian.math.logic.languages.regular.values import (
    DFA,
    MAX_COUNT_WORD_LENGTH,
    DFATransition,
)


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


def _dfa_full_alphabet_rejecting() -> DFA:
    """One non-accepting state looping on all 32 symbols."""

    return DFA(
        state_count=1,
        alphabet_size=32,
        transitions=tuple(
            DFATransition(source=0, symbol=symbol, target=0) for symbol in range(32)
        ),
        initial_state=0,
        accepting_states=(),
    )


def _dfa_rotating_binary(accepting_states: tuple[int, ...]) -> DFA:
    """A 64-state DFA whose symbols advance by zero or one state."""

    return DFA(
        state_count=64,
        alphabet_size=2,
        transitions=tuple(
            DFATransition(
                source=state,
                symbol=symbol,
                target=(state + symbol) % 64,
            )
            for state in range(64)
            for symbol in range(2)
        ),
        initial_state=0,
        accepting_states=accepting_states,
    )


def _dfa_accepting_only_zeros() -> DFA:
    """A 32-symbol DFA accepting exactly one word at every length."""

    return DFA(
        state_count=2,
        alphabet_size=32,
        transitions=tuple(
            DFATransition(
                source=state,
                symbol=symbol,
                target=0 if state == 0 and symbol == 0 else 1,
            )
            for state in range(2)
            for symbol in range(32)
        ),
        initial_state=0,
        accepting_states=(0,),
    )


def _dfa_cycle_with_transient_branching() -> DFA:
    """A 63-state accepting cycle with a 32-way branch only out of state 0.

    Every other cycle state continues on one symbol and sends the rest to a
    rejecting sink. Useful growth is ``32 ** ceil(length / 63)``, not ``32 ** length``.
    """

    cycle = 63
    sink = 63
    transitions = [
        DFATransition(source=0, symbol=symbol, target=1) for symbol in range(32)
    ]
    for state in range(1, cycle):
        nxt = (state + 1) % cycle
        transitions.append(DFATransition(source=state, symbol=0, target=nxt))
        transitions.extend(
            DFATransition(source=state, symbol=symbol, target=sink)
            for symbol in range(1, 32)
        )
    transitions.extend(
        DFATransition(source=sink, symbol=symbol, target=sink) for symbol in range(32)
    )
    return DFA(
        state_count=64,
        alphabet_size=32,
        transitions=tuple(transitions),
        initial_state=0,
        accepting_states=tuple(range(cycle)),
    )


def _dfa_with_transient_branching() -> DFA:
    """A DFA accepting 32 words despite a large first-step branch."""

    return DFA(
        state_count=3,
        alphabet_size=32,
        transitions=tuple(
            DFATransition(
                source=state,
                symbol=symbol,
                target=(1 if state == 0 or (state == 1 and symbol == 0) else 2),
            )
            for state in range(3)
            for symbol in range(32)
        ),
        initial_state=0,
        accepting_states=(1,),
    )


def _dfa_ternary_accepting() -> DFA:
    """One accepting state looping on three symbols."""

    return DFA(
        state_count=1,
        alphabet_size=3,
        transitions=tuple(
            DFATransition(source=0, symbol=symbol, target=0) for symbol in range(3)
        ),
        initial_state=0,
        accepting_states=(0,),
    )


def _dfa_binary_toggle() -> DFA:
    """Both symbols swap two states; only the initial state accepts."""

    return DFA(
        state_count=2,
        alphabet_size=2,
        transitions=(
            DFATransition(source=0, symbol=0, target=1),
            DFATransition(source=0, symbol=1, target=1),
            DFATransition(source=1, symbol=0, target=0),
            DFATransition(source=1, symbol=1, target=0),
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


def test_count_uses_flint_powering_above_the_previous_length_ceiling() -> None:
    dfa = _dfa_full_alphabet_accepting()
    result = compute_count(CountRequest(dfa=dfa, word_length=1_000))

    assert result.count == str(32**1_000)


def test_count_rejects_projected_result_digits_before_powering() -> None:
    with pytest.raises(OperationDomainValidationError, match="result digit bound"):
        count_accepted_words(_dfa_full_alphabet_accepting(), 22_000)


def test_count_admits_value_just_below_result_digit_bound() -> None:
    assert count_accepted_words(_dfa_full_alphabet_accepting(), 21_761) == 32**21_761


def test_count_prunes_rejecting_growth_before_admission() -> None:
    assert count_accepted_words(_dfa_accepting_only_zeros(), 22_000) == 1


def test_count_admits_transient_branching_before_sparse_tail() -> None:
    assert count_accepted_words(_dfa_with_transient_branching(), 22_000) == 32


def test_count_admits_cycle_transient_branching_from_path_sensitive_work() -> None:
    """Length 100000 on the 63-cycle is 32**1588 (2391 digits), not 32**100000."""

    length = 100_000
    expected = 32 ** ((length + 62) // 63)
    assert len(str(expected)) == 2391
    count = count_accepted_words(_dfa_cycle_with_transient_branching(), length)
    assert count == expected


def test_count_uses_tight_alphabet_power_digit_bound() -> None:
    """3**60000 has 28,628 digits; a ceil(log2) estimate would reject it."""

    assert count_accepted_words(_dfa_ternary_accepting(), 60_000) == 3**60_000


def test_count_empty_accepting_set_short_circuits_before_result_bound() -> None:
    """Empty accepting sets are exactly zero without charging large-n growth."""

    dfa = _dfa_full_alphabet_rejecting()
    assert compute_count(CountRequest(dfa=dfa, word_length=22_000)).count == "0"
    assert count_accepted_words(dfa, 22_000) == 0
    assert count_accepted_words(dfa, 0) == 0


def test_count_toggle_dfa_parity_is_exact() -> None:
    dfa = _dfa_binary_toggle()
    assert count_accepted_words(dfa, 1) == 0
    assert count_accepted_words(dfa, 2) == 4
    assert count_accepted_words(dfa, 3) == 0
    assert count_accepted_words(dfa, 10) == 2**10


def test_count_rejects_toggle_dfa_intermediate_explosion() -> None:
    """Odd max-length toggle counts are 0, but FLINT off-diagonals are 2**n."""

    with pytest.raises(OperationDomainValidationError, match="intermediate"):
        count_accepted_words(_dfa_binary_toggle(), MAX_COUNT_WORD_LENGTH)


def test_count_rejects_large_state_powering_work() -> None:
    with pytest.raises(OperationDomainValidationError, match="matrix powering"):
        count_accepted_words(_dfa_rotating_binary((0,)), 10_000)


def test_count_admits_large_state_powering_within_work_bound() -> None:
    dfa = _dfa_rotating_binary(tuple(range(64)))

    assert count_accepted_words(dfa, 5_000) == 2**5_000


def test_count_accepts_maximum_transport_exponent_for_compact_unary_dfa() -> None:
    state_count = 64
    dfa = DFA(
        state_count=state_count,
        alphabet_size=1,
        transitions=tuple(
            DFATransition(source=state, symbol=0, target=(state + 1) % state_count)
            for state in range(state_count)
        ),
        initial_state=0,
        accepting_states=(MAX_COUNT_WORD_LENGTH % state_count,),
    )

    assert count_accepted_words(dfa, MAX_COUNT_WORD_LENGTH) == 1


def test_count_request_rejects_exponent_above_transport_range() -> None:
    with pytest.raises(ValidationError):
        CountRequest(
            dfa=_dfa_even_zeros(),
            word_length=MAX_COUNT_WORD_LENGTH + 1,
        )


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
