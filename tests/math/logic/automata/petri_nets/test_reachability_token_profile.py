"""Exact token extrema over small independently enumerated Petri reachability graphs."""

from collections import deque

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.automata.petri_nets._models import (
    ReachabilityRequest,
)
from jacobian.math.logic.automata.petri_nets._tools import compute_reachability
from jacobian.math.logic.automata.petri_nets.profiles._models import (
    ReachabilityTokenProfileRequest,
)
from jacobian.math.logic.automata.petri_nets.profiles._tools import (
    compute_reachability_token_profile,
)
from jacobian.math.logic.automata.petri_nets.values import Marking, PetriNet


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
