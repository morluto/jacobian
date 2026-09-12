"""Exact accepted-word symbol Parikh profiles."""

import json
from itertools import product
from math import comb
from typing import Any, cast

import pytest
from tests.fixtures.accounting import assert_charged_work_parity

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.languages.regular._symbol_parikh import (
    MAX_SYMBOL_PARIKH_DP_WORK,
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


def test_profile_result_rejects_forged_json_cells() -> None:
    result = symbol_parikh_profile(
        SymbolParikhProfileRequest(dfa=ending_in_one(), word_length=3)
    )
    payload = result.model_dump(mode="json")
    payload["cells"][0]["symbol_counts"] = [0, 4]

    with pytest.raises(ValueError, match="nonnegative and sum"):
        SymbolParikhProfileResult.model_validate_json(json.dumps(payload))


def test_large_accepted_profile_uses_trusted_result_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    import jacobian.math.logic.languages.regular._symbol_parikh as profile

    calls = 0
    builtin_sorted = sorted

    def counted_sorted(*args: Any, **kwargs: Any) -> object:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("profile result construction replayed cell sorting")
        return builtin_sorted(*args, **kwargs)

    monkeypatch.setattr(profile, "sorted", cast(Any, counted_sorted), raising=False)
    result = symbol_parikh_profile(SymbolParikhProfileRequest(dfa=dfa, word_length=3))

    assert len(result.cells) == 5_984
    assert result.total_accepted_words == alphabet_size**3
    assert calls == 1


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


def test_symbol_profile_merges_many_distinct_transition_signatures() -> None:
    state_count = 8
    transitions = tuple(
        DFATransition(
            source=source,
            symbol=symbol,
            target=(2 * source + symbol) % state_count,
        )
        for source in range(state_count)
        for symbol in range(2)
    )
    dfa = DFA(
        state_count=state_count,
        alphabet_size=2,
        transitions=transitions,
        initial_state=0,
        accepting_states=tuple(range(state_count)),
    )

    result = symbol_parikh_profile(SymbolParikhProfileRequest(dfa=dfa, word_length=10))
    transition_signatures: set[tuple[int, ...]] = set()
    for word in product(range(2), repeat=10):
        state = dfa.initial_state
        signature = [0] * len(transitions)
        for symbol in word:
            transition_index = 2 * state + symbol
            signature[transition_index] += 1
            state = transitions[transition_index].target
        transition_signatures.add(tuple(signature))

    assert len(result.cells) == 11
    assert len(transition_signatures) == 617


def test_final_layer_scan_charges_every_reachable_state() -> None:
    dfa = DFA(
        state_count=3,
        alphabet_size=3,
        transitions=tuple(
            DFATransition(
                source=source,
                symbol=symbol,
                target=(source + symbol + 1) % 3,
            )
            for source in range(3)
            for symbol in range(3)
        ),
        initial_state=0,
        accepting_states=(0, 1, 2),
    )

    with pytest.raises(
        OperationResourceAdmissionError,
        match="symbol-Parikh DP or output exceeds",
    ):
        symbol_parikh_profile(SymbolParikhProfileRequest(dfa=dfa, word_length=75))


def _source_sensitive_dfa(
    reachable_state_count: int,
    *,
    state_count: int = 13,
    alphabet_size: int = 5,
) -> DFA:
    return DFA(
        state_count=state_count,
        alphabet_size=alphabet_size,
        transitions=tuple(
            DFATransition(
                source=source,
                symbol=symbol,
                target=(
                    source + 1
                    if symbol == 0 and source < reachable_state_count - 1
                    else 0
                    if symbol == 0
                    else source
                ),
            )
            for source in range(state_count)
            for symbol in range(alphabet_size)
        ),
        initial_state=0,
        accepting_states=tuple(range(state_count)),
    )


def test_profile_preserves_cheap_unreachable_state_case() -> None:
    result = symbol_parikh_profile(
        SymbolParikhProfileRequest(
            dfa=_source_sensitive_dfa(reachable_state_count=11),
            word_length=13,
        )
    )

    assert len(result.cells) == comb(17, 4)
    assert result.total_accepted_words == 5**13


def test_profile_preserves_49_reachable_state_case() -> None:
    result = symbol_parikh_profile(
        SymbolParikhProfileRequest(
            dfa=_source_sensitive_dfa(
                reachable_state_count=49,
                state_count=64,
                alphabet_size=15,
            ),
            word_length=3,
        )
    )

    assert len(result.cells) == comb(17, 14)
    assert result.total_accepted_words == 15**3


def test_transition_index_charge_rejects_before_indexing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reachable_state_count = 14
    state_count = 56
    alphabet_size = 16
    dfa = _source_sensitive_dfa(
        reachable_state_count=reachable_state_count,
        state_count=state_count,
        alphabet_size=alphabet_size,
    )
    length = 4
    transition_count = dfa.state_count * dfa.alphabet_size
    extension_cells = 0
    possible_word_count = 1
    for step in range(length):
        extension_cells += min(
            reachable_state_count * comb(step + alphabet_size - 1, alphabet_size - 1),
            possible_word_count,
        )
        possible_word_count *= alphabet_size
    output_materialization_cells = min(
        reachable_state_count * comb(length + alphabet_size - 1, alphabet_size - 1),
        possible_word_count,
    )
    without_index_work = (
        extension_cells * alphabet_size * max(1, alphabet_size)
        + output_materialization_cells * max(1, alphabet_size)
        + reachable_state_count * transition_count
    )
    assert without_index_work <= MAX_SYMBOL_PARIKH_DP_WORK
    assert without_index_work + transition_count > MAX_SYMBOL_PARIKH_DP_WORK

    import jacobian.math.logic.languages.regular._symbol_parikh as profile

    def fail(*_args: object, **_kwargs: object) -> dict[tuple[int, int], int]:
        raise AssertionError("transition index built before admission")

    monkeypatch.setattr(profile, "_build_transition_index", fail)
    with pytest.raises(
        OperationResourceAdmissionError,
        match="symbol-Parikh DP or output exceeds",
    ):
        profile.symbol_parikh_profile(
            SymbolParikhProfileRequest(dfa=dfa, word_length=length)
        )


def test_near_envelope_profile_execution_matches_admission_charge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.logic.languages.regular._symbol_parikh as profile

    dfa = _source_sensitive_dfa(
        reachable_state_count=62,
        state_count=62,
        alphabet_size=31,
    )
    length = 3
    alphabet_size = dfa.alphabet_size
    transition_count = dfa.state_count * alphabet_size
    output_bound = comb(length + alphabet_size - 1, alphabet_size - 1)
    reachable_count = 62

    extension_cells = 0
    possible_word_count = 1
    for step in range(length):
        extension_cells += min(
            reachable_count * comb(step + alphabet_size - 1, alphabet_size - 1),
            possible_word_count,
        )
        possible_word_count *= alphabet_size
    output_materialization_cells = min(
        reachable_count * output_bound,
        possible_word_count,
    )

    executed = {
        "transition_index": 0,
        "reachability_scan": 0,
        "extension_coordinate": 0,
        "output_materialization": 0,
    }
    original_index = profile._build_transition_index
    original_reachable = profile._reachable_states_without_index
    original_extend = profile._extend_profile_layer
    original_collect = profile._collect_profile

    def count_index(value: DFA) -> dict[tuple[int, int], int]:
        executed["transition_index"] += len(value.transitions)
        return original_index(value)

    def count_reachable(value: DFA) -> set[int]:
        result = original_reachable(value)
        executed["reachability_scan"] += len(result) * len(value.transitions)
        return result

    def count_extend(
        layer: dict[tuple[int, tuple[int, ...]], int],
        transitions: dict[tuple[int, int], int],
        reachable: set[int],
        size: int,
    ) -> dict[tuple[int, tuple[int, ...]], int]:
        executed["extension_coordinate"] += len(layer) * size * max(1, size)
        return original_extend(layer, transitions, reachable, size)

    def count_collect(
        layer: dict[tuple[int, tuple[int, ...]], int], accepting: set[int]
    ) -> dict[tuple[int, ...], int]:
        executed["output_materialization"] += len(layer) * max(1, alphabet_size)
        return original_collect(layer, accepting)

    monkeypatch.setattr(profile, "_build_transition_index", count_index)
    monkeypatch.setattr(profile, "_reachable_states_without_index", count_reachable)
    monkeypatch.setattr(profile, "_extend_profile_layer", count_extend)
    monkeypatch.setattr(profile, "_collect_profile", count_collect)

    result = profile.symbol_parikh_profile(
        profile.SymbolParikhProfileRequest(dfa=dfa, word_length=length)
    )

    charged = {
        "transition_index": transition_count,
        "reachability_scan": reachable_count * transition_count,
        "extension_coordinate": extension_cells * alphabet_size * max(1, alphabet_size),
        "output_materialization": output_materialization_cells * max(1, alphabet_size),
    }
    assert result.total_accepted_words == alphabet_size**length
    assert executed["transition_index"] == transition_count
    assert all(executed.values())
    assert sum(charged.values()) == MAX_SYMBOL_PARIKH_DP_WORK - 1_120
    assert_charged_work_parity(charged=charged, executed=executed)


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


def test_commuting_counter_dfa_admits_length_408_profile() -> None:
    state_count = 6
    dfa = DFA(
        state_count=state_count,
        alphabet_size=2,
        transitions=tuple(
            DFATransition(
                source=source,
                symbol=symbol,
                target=source if symbol == 0 else (source + 1) % state_count,
            )
            for source in range(state_count)
            for symbol in range(2)
        ),
        initial_state=0,
        accepting_states=tuple(range(state_count)),
    )
    result = symbol_parikh_profile(
        SymbolParikhProfileRequest(dfa=dfa, word_length=408)
    )
    assert result.total_accepted_words == 2**408
    assert len(result.cells) == 409
