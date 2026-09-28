"""Reachable dead-marking profiles respect bounded exploration cuts."""

from jacobian.math.logic.automata.petri_nets import (
    Marking,
    PetriNet,
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
    assert result.dead_markings == ((0, 0),)
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

    assert result.dead_markings == ((0,),)
    assert result.truncated is False
