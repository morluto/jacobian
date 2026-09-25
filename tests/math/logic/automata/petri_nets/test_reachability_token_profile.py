"""Exact token extrema over small independently enumerated Petri reachability graphs."""

from collections import deque

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.automata.petri_nets._models import (
    ReachabilityRequest,
    ReachabilityResult,
)
from jacobian.math.logic.automata.petri_nets._tools import compute_reachability
from jacobian.math.logic.automata.petri_nets.profiles._models import (
    ReachabilityTokenProfileRequest,
)
from jacobian.math.logic.automata.petri_nets.profiles._tools import (
    compute_reachability_token_profile,
)
from jacobian.math.logic.automata.petri_nets.profiles.operations import (
    reachability_token_profile,
)
from jacobian.math.logic.automata.petri_nets.values import (
    Marking,
    PetriMarkingState,
    PetriNet,
    PetriReachabilityEdge,
)


def _oracle(net: PetriNet, initial: tuple[int, ...]) -> list[tuple[int, ...]]:
    states = [initial]
    seen = {initial}
    queue = deque([initial])
    while queue:
        marking = queue.popleft()
        for transition in range(net.transition_count):
            if any(marking[p] < net.pre[p][transition] for p in range(net.place_count)):
                continue
            successor = tuple(
                marking[p] - net.pre[p][transition] + net.post[p][transition]
                for p in range(net.place_count)
            )
            if successor not in seen:
                seen.add(successor)
                states.append(successor)
                queue.append(successor)
    return states


@pytest.mark.parametrize(
    ("net", "initial"),
    [
        (
            PetriNet(
                place_count=2,
                transition_count=2,
                pre=((1, 0), (0, 1)),
                post=((0, 1), (1, 0)),
            ),
            (1, 0),
        ),
        (
            PetriNet(
                place_count=2,
                transition_count=2,
                pre=((1, 0), (0, 2)),
                post=((0, 1), (2, 0)),
            ),
            (1, 0),
        ),
    ],
)
def test_complete_profile_matches_independent_bfs_oracle(
    net: PetriNet, initial: tuple[int, ...]
) -> None:
    graph = compute_reachability(
        ReachabilityRequest(
            net=net, initial_marking=Marking(tokens=initial), max_states=16
        )
    )
    oracle = _oracle(net, initial)
    result = compute_reachability_token_profile(
        ReachabilityTokenProfileRequest(source_graph=graph)
    )
    assert result.completeness == "COMPLETE"
    assert [state.marking.tokens for state in graph.states] == oracle
    for place, profile in enumerate(result.place_ranges):
        values = [marking[place] for marking in oracle]
        assert (profile.minimum, profile.maximum) == (min(values), max(values))
        assert oracle[profile.minimum_state][place] == profile.minimum
        assert oracle[profile.maximum_state][place] == profile.maximum
    totals = [sum(marking) for marking in oracle]
    assert (result.total_minimum, result.total_maximum) == (min(totals), max(totals))
    assert oracle[result.total_minimum_state] == next(
        m for m in oracle if sum(m) == min(totals)
    )
    assert oracle[result.total_maximum_state] == next(
        m for m in oracle if sum(m) == max(totals)
    )


def test_truncated_profile_is_explicitly_only_observed_prefix() -> None:
    net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((0,),),
        post=((1,),),
    )
    graph = compute_reachability(
        ReachabilityRequest(net=net, initial_marking=Marking(tokens=(0,)), max_states=2)
    )
    assert graph.truncated
    result = compute_reachability_token_profile(
        ReachabilityTokenProfileRequest(source_graph=graph)
    )
    assert result.completeness == "OBSERVED_PREFIX"
    assert (result.place_ranges[0].minimum, result.place_ranges[0].maximum) == (0, 1)
    assert result.source_graph == graph


def test_false_complete_graph_cannot_authorize_full_reachable_extrema() -> None:
    net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((0,),),
        post=((0,),),
    )
    graph = compute_reachability(
        ReachabilityRequest(net=net, initial_marking=Marking(tokens=(0,)), max_states=2)
    )
    forged = graph.model_copy(update={"edges": ()})
    with pytest.raises(OperationDomainValidationError, match="every enabled successor"):
        compute_reachability_token_profile(
            ReachabilityTokenProfileRequest(source_graph=forged)
        )


def test_tied_extrema_use_lowest_source_state_index_not_bfs_history() -> None:
    net = PetriNet(
        place_count=3,
        transition_count=2,
        pre=((1, 1), (1, 0), (0, 1)),
        post=((0, 0), (0, 0), (0, 0)),
    )
    graph = compute_reachability(
        ReachabilityRequest(net=net, initial_marking=Marking(tokens=(1, 1, 1)))
    )
    assert tuple(state.marking.tokens for state in graph.states) == (
        (1, 1, 1),
        (0, 0, 1),
        (0, 1, 0),
    )
    # Swap the tied dead markings while preserving the initial state and every
    # firing edge. This is valid, but its state indices are not BFS discovery order.
    reordered = ReachabilityResult.model_construct(
        net=graph.net,
        initial_marking=graph.initial_marking,
        max_states=graph.max_states,
        states=(
            graph.states[0],
            PetriMarkingState(
                state_index=1, place_axis=(0, 1, 2), marking=graph.states[2].marking
            ),
            PetriMarkingState(
                state_index=2, place_axis=(0, 1, 2), marking=graph.states[1].marking
            ),
        ),
        edges=(
            PetriReachabilityEdge(source_state=0, transition=0, target_state=2),
            PetriReachabilityEdge(source_state=0, transition=1, target_state=1),
        ),
        truncated=False,
    )
    result = reachability_token_profile(reordered)
    assert result.total_minimum == 1
    assert result.total_minimum_state == 1
    assert result.source_graph.states[1].marking.tokens == (0, 1, 0)


def test_zero_place_net_has_one_complete_empty_state_profile() -> None:
    net = PetriNet(place_count=0, transition_count=0, pre=(), post=())
    graph = compute_reachability(
        ReachabilityRequest(net=net, initial_marking=Marking(tokens=()))
    )
    result = compute_reachability_token_profile(
        ReachabilityTokenProfileRequest(source_graph=graph)
    )
    assert result.completeness == "COMPLETE"
    assert result.place_ranges == ()
    assert (result.total_minimum, result.total_minimum_state) == (0, 0)
    assert (result.total_maximum, result.total_maximum_state) == (0, 0)


def test_oversized_arc_integer_is_rejected_before_output_stringification() -> None:
    valid_net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((0,),),
        post=((0,),),
    )
    forged_net = valid_net.model_copy(update={"pre": ((10**5000,),)})
    graph = ReachabilityResult.model_construct(
        net=forged_net,
        initial_marking=Marking(tokens=(0,)),
        max_states=1,
        states=(
            PetriMarkingState(
                state_index=0, place_axis=(0,), marking=Marking(tokens=(0,))
            ),
        ),
        edges=(),
        truncated=False,
    )
    with pytest.raises(OperationDomainValidationError, match="bounded nonnegative"):
        reachability_token_profile(graph)
