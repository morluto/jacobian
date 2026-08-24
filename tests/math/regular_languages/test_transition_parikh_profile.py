from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import product

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json, parse_canonical_integer
from jacobian.math.regular_languages._models import (
    TransitionParikhProfileRequest,
)
from jacobian.math.regular_languages._operations import (
    compute_transition_parikh_profile,
)
from jacobian.math.regular_languages.operations import (
    _MAX_TRANSITION_PROFILE_RESULT_BYTES,
    dfa_transition_carrier,
    transition_parikh_profile,
)
from jacobian.math.regular_languages.values import (
    DFA,
    MAX_LABELED_AUTOMATON_STATES,
    MAX_LABELED_AUTOMATON_TRANSITIONS,
    MAX_TRANSITION_PROFILE_PATH_LENGTH,
    AutomatonTransition,
    DFATransition,
    FiniteLabeledAutomaton,
    TransitionParikhProfile,
)


def _automaton(
    state_count: int,
    alphabet_size: int,
    rows: tuple[tuple[int, int, int], ...],
) -> FiniteLabeledAutomaton:
    return FiniteLabeledAutomaton(
        state_count=state_count,
        alphabet_size=alphabet_size,
        transitions=tuple(
            AutomatonTransition(
                transition_id=transition_id,
                source=source,
                symbol=symbol,
                target=target,
            )
            for transition_id, (source, symbol, target) in enumerate(rows)
        ),
    )


def _profile_map(profile: TransitionParikhProfile) -> Counter[tuple[int, ...]]:
    return Counter(
        {
            entry.transition_counts: parse_canonical_integer(entry.multiplicity)
            for entry in profile.entries
        }
    )


def _path_id_sequences(
    automaton: FiniteLabeledAutomaton,
    source_state: int,
    target_state: int,
    path_length: int,
) -> tuple[tuple[int, ...], ...]:
    layer = ((source_state, ()),)
    for _ in range(path_length):
        layer = tuple(
            (transition.target, (*path, transition.transition_id))
            for state, path in layer
            for transition in automaton.transitions
            if transition.source == state
        )
    return tuple(path for state, path in layer if state == target_state)


def _brute_profile(
    automaton: FiniteLabeledAutomaton,
    source_state: int,
    target_state: int,
    path_length: int,
) -> Counter[tuple[int, ...]]:
    histogram: Counter[tuple[int, ...]] = Counter()
    for path in _path_id_sequences(automaton, source_state, target_state, path_length):
        counts = [0] * len(automaton.transitions)
        for transition_id in path:
            counts[transition_id] += 1
        histogram[tuple(counts)] += 1
    return histogram


def test_two_loop_paths_form_a_histogram_not_a_set() -> None:
    automaton = _automaton(
        1,
        2,
        (
            (0, 0, 0),
            (0, 1, 0),
        ),
    )

    profile = transition_parikh_profile(automaton, 0, 0, 2)

    assert _profile_map(profile) == Counter(
        {
            (0, 2): 1,
            (1, 1): 2,
            (2, 0): 1,
        }
    )
    assert profile.total_path_count == "4"


def test_length_zero_and_empty_transition_conventions_retain_the_carrier() -> None:
    automaton = _automaton(2, 0, ())

    identity = transition_parikh_profile(automaton, 0, 0, 0)
    distinct = transition_parikh_profile(automaton, 0, 1, 0)
    positive_length = transition_parikh_profile(automaton, 0, 0, 1)

    assert identity.automaton == automaton
    assert _profile_map(identity) == Counter({(): 1})
    assert identity.total_path_count == "1"
    assert distinct.entries == ()
    assert distinct.total_path_count == "0"
    assert positive_length.entries == ()
    assert positive_length.total_path_count == "0"


def test_profile_multiplicities_sum_to_independent_state_path_count() -> None:
    automaton = _automaton(
        3,
        2,
        (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 1),
            (1, 0, 0),
            (1, 1, 2),
            (2, 0, 2),
        ),
    )
    path_length = 5
    state_counts = [0] * automaton.state_count
    state_counts[0] = 1
    for _ in range(path_length):
        next_counts = [0] * automaton.state_count
        for transition in automaton.transitions:
            next_counts[transition.target] += state_counts[transition.source]
        state_counts = next_counts

    profile = transition_parikh_profile(automaton, 0, 2, path_length)

    assert sum(_profile_map(profile).values()) == state_counts[2]
    assert parse_canonical_integer(profile.total_path_count) == state_counts[2]


def test_profile_satisfies_the_transition_recurrence() -> None:
    automaton = _automaton(
        2,
        2,
        (
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 1, 1),
        ),
    )
    path_length = 4
    expected: Counter[tuple[int, ...]] = Counter()
    for transition in automaton.transitions:
        if transition.target != 1:
            continue
        prefix = transition_parikh_profile(
            automaton, 0, transition.source, path_length - 1
        )
        for vector, multiplicity in _profile_map(prefix).items():
            extended = list(vector)
            extended[transition.transition_id] += 1
            expected[tuple(extended)] += multiplicity

    assert _profile_map(transition_parikh_profile(automaton, 0, 1, path_length)) == (
        expected
    )


def test_transition_counts_project_to_symbol_parikh_counts() -> None:
    automaton = _automaton(
        2,
        2,
        (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 1),
            (1, 0, 0),
            (1, 1, 1),
        ),
    )
    profile = transition_parikh_profile(automaton, 0, 1, 3)
    projected: Counter[tuple[int, ...]] = Counter()
    for vector, multiplicity in _profile_map(profile).items():
        symbol_counts = tuple(
            sum(
                vector[transition.transition_id]
                for transition in automaton.transitions
                if transition.symbol == symbol
            )
            for symbol in range(automaton.alphabet_size)
        )
        projected[symbol_counts] += multiplicity

    brute: Counter[tuple[int, ...]] = Counter()
    for path in _path_id_sequences(automaton, 0, 1, 3):
        symbols = [
            automaton.transitions[transition_id].symbol for transition_id in path
        ]
        brute[tuple(symbols.count(symbol) for symbol in range(2))] += 1

    assert projected == brute


def test_state_relabeling_transports_the_profile_without_changing_its_axis() -> None:
    automaton = _automaton(
        3,
        2,
        (
            (0, 0, 1),
            (0, 1, 2),
            (1, 0, 2),
            (2, 1, 0),
        ),
    )
    relabel = (2, 0, 1)
    transported = _automaton(
        3,
        2,
        tuple(
            (relabel[transition.source], transition.symbol, relabel[transition.target])
            for transition in automaton.transitions
        ),
    )

    original_profile = transition_parikh_profile(automaton, 0, 2, 4)
    transported_profile = transition_parikh_profile(
        transported, relabel[0], relabel[2], 4
    )

    assert original_profile.entries == transported_profile.entries
    assert original_profile.total_path_count == transported_profile.total_path_count


def test_transition_relabeling_permutes_profile_coordinates_canonically() -> None:
    automaton = _automaton(
        2,
        2,
        (
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 1, 1),
        ),
    )
    permutation = (2, 0, 3, 1)
    relabeled = _automaton(
        automaton.state_count,
        automaton.alphabet_size,
        tuple(
            (
                automaton.transitions[old_id].source,
                automaton.transitions[old_id].symbol,
                automaton.transitions[old_id].target,
            )
            for old_id in permutation
        ),
    )

    original = _profile_map(transition_parikh_profile(automaton, 0, 1, 4))
    transported = _profile_map(transition_parikh_profile(relabeled, 0, 1, 4))

    assert transported == Counter(
        {
            tuple(vector[old_id] for old_id in permutation): multiplicity
            for vector, multiplicity in original.items()
        }
    )


def test_dfa_projection_supplies_the_canonical_carrier_unchanged() -> None:
    dfa = DFA(
        state_count=2,
        alphabet_size=2,
        transitions=(
            DFATransition(source=1, symbol=1, target=1),
            DFATransition(source=0, symbol=1, target=1),
            DFATransition(source=1, symbol=0, target=0),
            DFATransition(source=0, symbol=0, target=0),
        ),
        initial_state=0,
        accepting_states=(1,),
    )

    carrier = dfa_transition_carrier(dfa)
    profile = transition_parikh_profile(carrier, 0, 1, 2)

    assert tuple(
        (transition.transition_id, transition.source, transition.symbol)
        for transition in carrier.transitions
    ) == ((0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1))
    assert profile.automaton == carrier


@pytest.mark.property
@settings(max_examples=80, deadline=None)
@given(
    rows=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=2),
            st.integers(min_value=0, max_value=1),
            st.integers(min_value=0, max_value=2),
        ),
        max_size=4,
    ),
    source_state=st.integers(min_value=0, max_value=2),
    target_state=st.integers(min_value=0, max_value=2),
    path_length=st.integers(min_value=0, max_value=4),
)
def test_sparse_dp_matches_bounded_brute_force(
    rows: list[tuple[int, int, int]],
    source_state: int,
    target_state: int,
    path_length: int,
) -> None:
    automaton = _automaton(3, 2, tuple(rows))

    profile = transition_parikh_profile(
        automaton, source_state, target_state, path_length
    )

    assert _profile_map(profile) == _brute_profile(
        automaton, source_state, target_state, path_length
    )


def test_wire_result_rejects_source_and_conclusion_forgeries() -> None:
    automaton = _automaton(
        2,
        2,
        (
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
        ),
    )
    result = compute_transition_parikh_profile(
        TransitionParikhProfileRequest(
            automaton=automaton,
            source_state=0,
            target_state=1,
            path_length=1,
        )
    )
    payload = result.model_dump(mode="json")
    forgeries: list[dict[str, object]] = []

    wrong_vector = deepcopy(payload)
    wrong_vector["entries"][0]["transition_counts"] = [1, 0, 0]  # type: ignore[index]
    forgeries.append(wrong_vector)

    wrong_multiplicity = deepcopy(payload)
    wrong_multiplicity["entries"][0]["multiplicity"] = "2"  # type: ignore[index]
    wrong_multiplicity["total_path_count"] = "2"
    forgeries.append(wrong_multiplicity)

    wrong_source = deepcopy(payload)
    wrong_source["source_state"] = 1
    forgeries.append(wrong_source)

    wrong_target = deepcopy(payload)
    wrong_target["target_state"] = 0
    forgeries.append(wrong_target)

    wrong_length = deepcopy(payload)
    wrong_length["path_length"] = 2
    forgeries.append(wrong_length)

    wrong_transition = deepcopy(payload)
    wrong_transition["automaton"]["transitions"][1]["target"] = 0  # type: ignore[index]
    forgeries.append(wrong_transition)

    for forged in forgeries:
        with pytest.raises(ValidationError):
            TransitionParikhProfile.model_validate(forged)


def test_transition_axis_requires_contiguous_stable_identifiers() -> None:
    with pytest.raises(ValidationError, match="contiguous zero-based axis"):
        FiniteLabeledAutomaton(
            state_count=1,
            alphabet_size=1,
            transitions=(
                AutomatonTransition(
                    transition_id=1,
                    source=0,
                    symbol=0,
                    target=0,
                ),
            ),
        )

    parallel = _automaton(1, 1, ((0, 0, 0), (0, 0, 0)))
    assert _profile_map(transition_parikh_profile(parallel, 0, 0, 1)) == Counter(
        {(0, 1): 1, (1, 0): 1}
    )


def test_request_rejects_out_of_range_endpoint_before_execution() -> None:
    with pytest.raises(ValidationError, match="source_state"):
        TransitionParikhProfileRequest(
            automaton=_automaton(1, 0, ()),
            source_state=1,
            target_state=0,
            path_length=0,
        )


def _loop_automaton(
    transition_count: int, *, state_count: int = 1
) -> FiniteLabeledAutomaton:
    return _automaton(
        state_count,
        1,
        tuple((0, 0, 0) for _ in range(transition_count)),
    )


def test_request_rejects_excessive_dp_work_even_when_target_is_unreachable() -> None:
    automaton = _loop_automaton(3, state_count=2)
    TransitionParikhProfileRequest(
        automaton=automaton,
        source_state=0,
        target_state=1,
        path_length=157,
    )
    with pytest.raises(ValidationError, match="DP transition-update bound"):
        TransitionParikhProfileRequest(
            automaton=automaton,
            source_state=0,
            target_state=1,
            path_length=158,
        )


def test_request_preflight_scans_only_reachable_outgoing_transitions() -> None:
    transition_count = MAX_LABELED_AUTOMATON_TRANSITIONS
    automaton = _automaton(
        2,
        1,
        ((0, 0, 0), *((1, 0, 1) for _ in range(transition_count - 1))),
    )

    TransitionParikhProfileRequest(
        automaton=automaton,
        source_state=0,
        target_state=0,
        path_length=1_000,
    )


def test_request_rejects_excessive_dense_vector_update_work() -> None:
    automaton = _loop_automaton(11, state_count=2)
    TransitionParikhProfileRequest(
        automaton=automaton,
        source_state=0,
        target_state=1,
        path_length=9,
    )
    with pytest.raises(ValidationError, match="dense-vector update-work bound"):
        TransitionParikhProfileRequest(
            automaton=automaton,
            source_state=0,
            target_state=1,
            path_length=10,
        )


def test_request_rejects_excessive_profile_cell_count() -> None:
    automaton = _loop_automaton(9)
    TransitionParikhProfileRequest(
        automaton=automaton,
        source_state=0,
        target_state=0,
        path_length=9,
    )
    with pytest.raises(ValidationError, match="profile-cell bound"):
        TransitionParikhProfileRequest(
            automaton=automaton,
            source_state=0,
            target_state=0,
            path_length=10,
        )


def test_request_rejects_excessive_intermediate_vector_coordinates() -> None:
    automaton = _loop_automaton(159, state_count=2)
    with pytest.raises(ValidationError, match="intermediate vector-coordinate bound"):
        TransitionParikhProfileRequest(
            automaton=automaton,
            source_state=0,
            target_state=1,
            path_length=2,
        )
    TransitionParikhProfileRequest(
        automaton=_loop_automaton(158, state_count=2),
        source_state=0,
        target_state=1,
        path_length=2,
    )


def test_request_rejects_excessive_serialized_result() -> None:
    accepted = TransitionParikhProfileRequest(
        automaton=_loop_automaton(129),
        source_state=0,
        target_state=0,
        path_length=2,
    )
    encoded = encode_strict_json(
        compute_transition_parikh_profile(accepted).model_dump(mode="json")
    )
    assert len(encoded) <= _MAX_TRANSITION_PROFILE_RESULT_BYTES

    with pytest.raises(ValidationError, match="serialized-result bound"):
        TransitionParikhProfileRequest(
            automaton=_loop_automaton(130),
            source_state=0,
            target_state=0,
            path_length=2,
        )


def test_length_bound_keeps_large_degenerate_cases_typed() -> None:
    automaton = _automaton(2, 0, ())
    request = TransitionParikhProfileRequest(
        automaton=automaton,
        source_state=0,
        target_state=1,
        path_length=MAX_TRANSITION_PROFILE_PATH_LENGTH,
    )
    assert compute_transition_parikh_profile(request).entries == ()

    with pytest.raises(ValidationError, match="less than or equal"):
        TransitionParikhProfileRequest(
            automaton=automaton,
            source_state=0,
            target_state=1,
            path_length=MAX_TRANSITION_PROFILE_PATH_LENGTH + 1,
        )


def _atlas_clock_carry_product_automaton() -> tuple[FiniteLabeledAutomaton, int]:
    """Atlas's three clock states in each role, crossed with carry."""

    delta = ((0, 0, 1), (0, 0, 2), (0, 0, 0))
    carries = (-1, 0, 1)
    states = tuple(
        (*clock_states, carry)
        for clock_states in product(range(3), repeat=3)
        for carry in carries
    )
    state_index = {state: index for index, state in enumerate(states)}
    rows: list[tuple[int, int, int]] = []
    for source_state, (left_state, middle_state, right_state, carry) in enumerate(
        states
    ):
        for symbol, (a, b, c) in enumerate(product(range(3), repeat=3)):
            value = a + c - 2 * b + carry
            if value % 3 == 0 and value // 3 in carries:
                target = (
                    delta[left_state][a],
                    delta[middle_state][b],
                    delta[right_state][c],
                    value // 3,
                )
                rows.append((source_state, symbol, state_index[target]))
    start = state_index[(0, 0, 0, 0)]
    return _automaton(len(states), 27, tuple(rows)), start


def test_atlas_length_three_clock_cubed_carry_profile_is_complete() -> None:
    # Source: route_enum.py and macro_engine.py at Atlas revision
    # 0394e3d3b249439ffabec7d96a3311aa441651b8. The source script enumerates
    # the full clock^3/carry product exactly only through length four.
    automaton, start = _atlas_clock_carry_product_automaton()

    profile = transition_parikh_profile(automaton, start, start, 3)

    assert automaton.state_count == 81
    assert len(automaton.transitions) == 729
    assert profile.total_path_count != "0"
    assert _profile_map(profile) == _brute_profile(automaton, start, start, 3)


def test_labeled_automaton_state_materialization_boundary() -> None:
    FiniteLabeledAutomaton(
        state_count=MAX_LABELED_AUTOMATON_STATES,
        alphabet_size=0,
        transitions=(),
    )
    with pytest.raises(ValidationError, match="less than or equal"):
        FiniteLabeledAutomaton(
            state_count=MAX_LABELED_AUTOMATON_STATES + 1,
            alphabet_size=0,
            transitions=(),
        )


def _clock_automaton(delta: tuple[tuple[int, int, int], ...]) -> FiniteLabeledAutomaton:
    return _automaton(
        3,
        3,
        tuple(
            (state, symbol, delta[state][symbol])
            for state in range(3)
            for symbol in range(3)
        ),
    )


def _states_reachable_through_length_three(
    automaton: FiniteLabeledAutomaton,
) -> set[int]:
    return {
        target
        for target in range(automaton.state_count)
        for length in range(4)
        if transition_parikh_profile(automaton, 0, target, length).total_path_count
        != "0"
    }


def test_atlas_nine_port_clock_is_unreachable_while_seven_port_clock_is_reachable() -> (
    None
):
    # PORT_CAPACITY_S3.json at the pinned Atlas revision records maxima 9 over
    # all ternary clocks and 7 after requiring every state reachable by L=3.
    nine_port_clock = _clock_automaton(((0, 0, 0), (0, 0, 0), (0, 0, 0)))
    seven_port_clock = _clock_automaton(((0, 0, 1), (0, 0, 2), (0, 0, 0)))

    assert (
        sum(transition.target == 0 for transition in nine_port_clock.transitions) == 9
    )
    assert _states_reachable_through_length_three(nine_port_clock) == {0}
    assert (
        sum(transition.target == 0 for transition in seven_port_clock.transitions) == 7
    )
    assert _states_reachable_through_length_three(seven_port_clock) == {0, 1, 2}
