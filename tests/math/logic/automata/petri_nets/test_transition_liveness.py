"""Exact transition liveness over complete finite Petri reachability graphs."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.automata.petri_nets.liveness.operations import (
    transition_liveness_profile,
)
from jacobian.math.logic.automata.petri_nets.operations import reachability_graph
from jacobian.math.logic.automata.petri_nets.values import Marking, PetriNet


def _independent_live_statuses(
    net: PetriNet, initial: tuple[int, ...]
) -> tuple[bool, ...]:
    """Tiny-net oracle: independently explore each source marking's future."""
    reachable = {initial}
    queue = [initial]
    while queue:
        marking = queue.pop()
        for transition in range(net.transition_count):
            if any(marking[p] < net.pre[p][transition] for p in range(net.place_count)):
                continue
            successor = tuple(
                marking[p] - net.pre[p][transition] + net.post[p][transition]
                for p in range(net.place_count)
            )
            if successor not in reachable:
                reachable.add(successor)
                queue.append(successor)
    statuses = []
    for transition in range(net.transition_count):
        live = True
        for source in reachable:
            future = {source}
            pending = [source]
            can_fire = False
            while pending and not can_fire:
                marking = pending.pop()
                for candidate in range(net.transition_count):
                    if any(
                        marking[p] < net.pre[p][candidate]
                        for p in range(net.place_count)
                    ):
                        continue
                    successor = tuple(
                        marking[p] - net.pre[p][candidate] + net.post[p][candidate]
                        for p in range(net.place_count)
                    )
                    if candidate == transition:
                        can_fire = True
                        break
                    if successor not in future:
                        future.add(successor)
                        pending.append(successor)
            if not can_fire:
                live = False
                break
        statuses.append(live)
    return tuple(statuses)


def test_distinguishes_liveness_from_firing_once_and_never_firing() -> None:
    net = PetriNet(
        place_count=1,
        transition_count=3,
        pre=((1, 0, 2),),
        post=((0, 0, 0),),
    )
    graph = reachability_graph(net, Marking(tokens=(1,)), max_states=4)
    result = transition_liveness_profile(graph)

    assert graph.truncated is False
    assert tuple(entry.status for entry in result.transitions) == (
        "NOT_LIVE",
        "LIVE",
        "NOT_LIVE",
    )
    assert tuple(entry.witness_state for entry in result.transitions) == (1, None, 0)
    assert tuple(entry.status == "LIVE" for entry in result.transitions) == (
        _independent_live_statuses(net, (1,))
    )


def test_truncated_graph_never_claims_liveness_from_an_observed_loop() -> None:
    net = PetriNet(
        place_count=1,
        transition_count=2,
        pre=((1, 0),),
        post=((0, 0),),
    )
    graph = reachability_graph(net, Marking(tokens=(1,)), max_states=1)
    result = transition_liveness_profile(graph)
    assert graph.truncated is True
    assert tuple(entry.status for entry in result.transitions) == ("UNKNOWN", "UNKNOWN")


def test_empty_transition_axis_and_zero_place_unconditional_transition() -> None:
    no_transitions = PetriNet(place_count=1, transition_count=0, pre=((),), post=((),))
    empty_profile = transition_liveness_profile(
        reachability_graph(no_transitions, Marking(tokens=(0,)), max_states=1)
    )
    assert empty_profile.transitions == ()

    zero_place = PetriNet(place_count=0, transition_count=1, pre=(), post=())
    zero_place_graph = reachability_graph(zero_place, Marking(tokens=()), max_states=1)
    assert transition_liveness_profile(zero_place_graph).transitions[0].status == "LIVE"


def test_rejects_forged_untruncated_graph_missing_enabled_successor() -> None:
    net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((1,),),
        post=((0,),),
    )
    graph = reachability_graph(net, Marking(tokens=(1,)), max_states=2)
    forged = graph.model_copy(update={"edges": (), "truncated": False})
    with pytest.raises(OperationDomainValidationError):
        transition_liveness_profile(forged)


def test_parent_matrix_work_is_not_charged_twice() -> None:
    """The shared preflight's fourth value is already total parent matrix work.

    It was multiplied by the source dimensions again, which squared the axis
    cost: a valid one-state 64x64 graph whose initial and state markings retain
    their net charged 33,558,528 units against a 1,000,000 limit and was refused
    by this operation while the sibling terminal-SCC analysis accepted it. The
    same graph needs only bounded validation, not 33M units.
    """
    from jacobian.math.logic.automata.petri_nets.operations import (
        _preflight_terminal_scc_graph_shape,
        reachability_terminal_scc_profile,
    )

    size = 64
    net = PetriNet(
        place_count=size,
        transition_count=size,
        pre=tuple((0,) * size for _ in range(size)),
        post=tuple((0,) * size for _ in range(size)),
    )
    marking = Marking(net=net, tokens=(1,) * size)
    graph = reachability_graph(net, marking, max_states=8)

    *_, parent_work = _preflight_terminal_scc_graph_shape(graph)
    # the preflight reports total parent matrix cells, not a marking count:
    # the initial marking and the single state each retain their 64x64 net
    assert parent_work == 2 * size * size

    # the liveness term used to be (1 + 1) * 64 * 64 * 64 * 64 = 33,558,528
    result = transition_liveness_profile(graph)
    assert len(result.transitions) == size

    # the sibling that already charged the value directly still agrees
    assert reachability_terminal_scc_profile(graph) is not None
