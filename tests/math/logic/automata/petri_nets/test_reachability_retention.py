"""Reachability consumers retain only the observations needed for their result."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.petri_nets import (
    Marking,
    PetriNet,
    operations,
    reachability_graph,
    reachable_dead_markings,
)
from jacobian.math.logic.automata.petri_nets.values import (
    MAX_PETRI_MARKING,
    MAX_REACHABILITY_FIRING_RECORDS,
)


def _growing_net(transitions: int) -> PetriNet:
    return PetriNet(
        place_count=1,
        transition_count=transitions,
        pre=((0,) * transitions,),
        post=((1,) * transitions,),
    )


def test_graph_exploration_retains_no_enabled_transition_families() -> None:
    """The graph's firing records must not have a second retained copy."""
    net = _growing_net(8)
    states, edges, truncated, unused_observations = operations._explore_reachability(
        net, Marking(net=net, tokens=(0,)), 16
    )
    assert states == [(index,) for index in range(16)]
    assert edges == [(index, t, index + 1) for index in range(15) for t in range(8)]
    assert truncated is True
    assert unused_observations == []


def test_graph_keeps_the_admitted_firing_record_boundary() -> None:
    net = _growing_net(50)
    max_states = MAX_REACHABILITY_FIRING_RECORDS // net.transition_count
    result = reachability_graph(net, Marking(net=net, tokens=(0,)), max_states)
    assert tuple(state.marking.tokens for state in result.states) == tuple(
        (index,) for index in range(MAX_PETRI_MARKING + 1)
    )
    assert tuple(
        (edge.source_state, edge.transition, edge.target_state) for edge in result.edges
    ) == tuple(
        (index, transition, index + 1)
        for index in range(MAX_PETRI_MARKING)
        for transition in range(net.transition_count)
    )
    assert result.truncated is True


def test_dead_exploration_retains_only_dead_state_indices() -> None:
    net = PetriNet(
        place_count=2,
        transition_count=2,
        pre=((1, 1), (0, 0)),
        post=((0, 0), (1, 2)),
    )
    initial = Marking(net=net, tokens=(1, 0))
    states, edges, truncated, dead_indices = operations._explore_reachability(
        net, initial, 4, collect_edges=False, collect_dead_states=True
    )
    assert states == [(1, 0), (0, 1), (0, 2)]
    assert edges == []
    assert truncated is False
    assert dead_indices == [1, 2]
    assert reachable_dead_markings(net, initial, 4).dead_markings == ((0, 1), (0, 2))


@pytest.mark.parametrize("max_states", (1, 3, 4))
def test_dead_markings_use_enabledness_once_and_respect_the_state_cut(
    max_states: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Count real enabling evaluations and check a known branching dead set."""
    net = PetriNet(
        place_count=2,
        transition_count=2,
        pre=((1, 1), (0, 0)),
        post=((0, 0), (1, 2)),
    )
    initial = Marking(net=net, tokens=(1, 0))
    original = operations._enabled_transition_indices
    visited: list[tuple[int, ...]] = []

    def observe(candidate: PetriNet, marking: Marking) -> list[int]:
        visited.append(marking.tokens)
        return original(candidate, marking)

    monkeypatch.setattr(operations, "_enabled_transition_indices", observe)
    result = reachable_dead_markings(net, initial, max_states)
    assert visited == ([(1, 0)] if max_states == 1 else [(1, 0), (0, 1), (0, 2)])
    assert result.dead_markings == (() if max_states == 1 else ((0, 1), (0, 2)))
    assert result.truncated is (max_states == 1)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_token_cut_does_not_create_a_dead_marking() -> None:
    net = _growing_net(8)
    initial = Marking(net=net, tokens=(MAX_PETRI_MARKING,))
    graph = reachability_graph(net, initial, 8)
    dead = reachable_dead_markings(net, initial, 8)
    assert len(graph.states) == 1
    assert graph.edges == ()
    assert graph.truncated is True
    assert dead.dead_markings == ()
    assert dead.truncated is True


@pytest.mark.parametrize("places,transitions", ((0, 0), (0, 2), (1, 0)))
def test_empty_axes_preserve_graph_and_dead_markings(
    places: int, transitions: int
) -> None:
    net = PetriNet(
        place_count=places,
        transition_count=transitions,
        pre=tuple((0,) * transitions for _ in range(places)),
        post=tuple((0,) * transitions for _ in range(places)),
    )
    initial = Marking(net=net, tokens=(0,) * places)
    graph = reachability_graph(net, initial, 100_000)
    dead = reachable_dead_markings(net, initial, 100_000)
    assert tuple(state.marking.tokens for state in graph.states) == (initial.tokens,)
    assert len(graph.edges) == transitions
    assert graph.truncated is False
    assert dead.dead_markings == (() if transitions else (initial.tokens,))
    assert dead.truncated is False


@pytest.mark.parametrize("consumer", (reachability_graph, reachable_dead_markings))
def test_firing_record_refusal_precedes_exploration(
    consumer: Callable[..., object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An over-budget request is refused before any enabling or firing work."""
    net = _growing_net(50)
    calls: list[str] = []

    def unexpected_exploration(*args: object, **kwargs: object) -> object:
        calls.append("exploration")
        raise AssertionError("admission must run before exploration")

    monkeypatch.setattr(operations, "_explore_reachability", unexpected_exploration)
    with pytest.raises(OperationResourceAdmissionError) as error:
        consumer(
            net,
            Marking(net=net, tokens=(0,)),
            MAX_REACHABILITY_FIRING_RECORDS // net.transition_count + 1,
        )
    assert error.value.errors()[0]["type"] == "petri_net.reachability_bound"
    assert calls == []
