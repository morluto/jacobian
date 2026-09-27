"""Reachable dead-marking profiles respect bounded exploration cuts."""

import jacobian.math.logic.automata.petri_nets.operations as petri_operations
from jacobian.math.logic.automata.petri_nets import (
    Marking,
    PetriNet,
    enabled_transitions,
    reachable_dead_markings,
)


def test_complete_exploration_returns_exact_reachable_dead_markings() -> None:
    net = PetriNet.model_validate(
        {
            "place_count": 2,
            "transition_count": 2,
            "pre": [[1, 0], [0, 1]],
            "post": [[0, 0], [1, 0]],
        }
    )
    result = reachable_dead_markings(net, Marking(tokens=(1, 0), net=net), 8)

    assert result.net == net
    assert result.initial_marking == Marking(tokens=(1, 0), net=net)
    assert result.dead_markings == (Marking(tokens=(0, 0), net=net),)
    assert enabled_transitions(net, result.dead_markings[0]).transitions == ()
    assert result.truncated is False


def test_truncated_enabled_state_is_not_misreported_as_dead() -> None:
    net = PetriNet.model_validate(
        {
            "place_count": 1,
            "transition_count": 1,
            "pre": [[1]],
            "post": [[0]],
        }
    )
    result = reachable_dead_markings(net, Marking(tokens=(1,)), 1)

    assert result.dead_markings == ()
    assert result.truncated is True


def test_initial_marking_can_itself_be_dead() -> None:
    net = PetriNet.model_validate(
        {
            "place_count": 1,
            "transition_count": 1,
            "pre": [[1]],
            "post": [[0]],
        }
    )
    result = reachable_dead_markings(net, Marking(tokens=(0,)), 4)

    assert result.dead_markings == (Marking(tokens=(0,), net=net),)
    assert result.truncated is False


def test_zero_transition_output_bound_uses_the_single_effective_state() -> None:
    net = PetriNet(
        place_count=64,
        transition_count=0,
        pre=((),) * 64,
        post=((),) * 64,
    )
    result = reachable_dead_markings(net, Marking(tokens=(0,) * 64), max_states=100_000)

    assert result.dead_markings == (Marking(tokens=(0,) * 64, net=net),)
    assert result.truncated is False


def test_reachability_records_deadness_during_the_original_bfs(
    monkeypatch,
) -> None:
    net = PetriNet(place_count=1, transition_count=1, pre=((1,),), post=((0,),))
    calls = 0
    original = petri_operations._enabled_transition_indices

    def count_enabled_checks(net_arg, marking):
        nonlocal calls
        calls += 1
        return original(net_arg, marking)

    monkeypatch.setattr(
        petri_operations, "_enabled_transition_indices", count_enabled_checks
    )
    result = reachable_dead_markings(net, Marking(tokens=(1,)), max_states=4)

    assert result.dead_markings == (Marking(tokens=(0,), net=net),)
    assert calls == 2
