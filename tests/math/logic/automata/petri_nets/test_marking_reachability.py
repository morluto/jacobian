"""Target-directed Petri marking reachability and bounded conclusions."""

from __future__ import annotations

from collections import deque

from jacobian.math.logic.automata.petri_nets import (
    Marking,
    PetriNet,
    marking_reachability,
    replay_firing_sequence,
)


def _cycle_net() -> PetriNet:
    return PetriNet.model_validate(
        {
            "place_count": 2,
            "transition_count": 2,
            "pre": [[1, 0], [0, 1]],
            "post": [[0, 1], [1, 0]],
        }
    )


def _reference_closure(net: PetriNet, initial: tuple[int, ...]) -> set[tuple[int, ...]]:
    """Small independent executable-semantics oracle for bounded fixtures."""
    reached = {initial}
    queue = deque([initial])
    while queue:
        marking = queue.popleft()
        for transition in range(net.transition_count):
            if any(
                marking[place] < net.pre[place][transition]
                for place in range(net.place_count)
            ):
                continue
            successor = tuple(
                marking[place]
                - net.pre[place][transition]
                + net.post[place][transition]
                for place in range(net.place_count)
            )
            if successor not in reached:
                reached.add(successor)
                queue.append(successor)
    return reached


def test_found_sequence_replays_to_the_bound_target() -> None:
    net = _cycle_net()
    initial = Marking(tokens=(1, 0), net=net)
    target = Marking(tokens=(0, 1), net=net)

    result = marking_reachability(net, initial, target, max_states=2)

    assert result.status == "REACHABLE"
    assert result.net == net
    assert result.initial_marking == initial
    assert result.target_marking == target
    assert result.sequence is not None
    replay = replay_firing_sequence(net, initial, result.sequence.transitions)
    assert replay.status == "FIRES"
    assert replay.final_marking == target
    assert replay.state_equation_residual == (0, 0)


def test_small_exhaustive_queries_match_independent_firing_closure() -> None:
    net = _cycle_net()
    markings = tuple((a, b) for a in range(3) for b in range(3))

    for source in markings:
        closure = _reference_closure(net, source)
        for target in markings:
            result = marking_reachability(
                net,
                Marking(tokens=source, net=net),
                Marking(tokens=target, net=net),
                max_states=10,
            )
            assert result.status == (
                "REACHABLE" if target in closure else "UNREACHABLE"
            )
            if target in closure:
                assert result.sequence is not None
                replay = replay_firing_sequence(
                    net, Marking(tokens=source), result.sequence.transitions
                )
                assert replay.status == "FIRES"
                assert replay.final_marking.tokens == target
            else:
                assert result.sequence is None
                assert result.incomplete_reasons == ()


def test_identity_target_returns_the_empty_sequence_even_with_a_cycle() -> None:
    net = _cycle_net()
    marking = Marking(tokens=(1, 0), net=net)

    result = marking_reachability(net, marking, marking, max_states=1)

    assert result.status == "REACHABLE"
    assert result.sequence is not None
    assert result.sequence.transitions == ()
    assert result.explored_state_count == 1


def test_state_limit_cut_is_incomplete_not_unreachable() -> None:
    net = PetriNet.model_validate(
        {
            "place_count": 3,
            "transition_count": 2,
            "pre": [[1, 0], [0, 1], [0, 0]],
            "post": [[0, 0], [1, 0], [0, 1]],
        }
    )
    result = marking_reachability(
        net, Marking(tokens=(1, 0, 0)), Marking(tokens=(0, 0, 1)), max_states=2
    )

    assert result.status == "INCOMPLETE"
    assert result.sequence is None
    assert result.incomplete_reasons == ("STATE_LIMIT",)
    assert result.explored_state_count == 2


def test_token_envelope_cut_is_incomplete_not_unreachable() -> None:
    net = PetriNet.model_validate(
        {
            "place_count": 1,
            "transition_count": 1,
            "pre": [[0]],
            "post": [[1]],
        }
    )
    result = marking_reachability(
        net, Marking(tokens=(1000,)), Marking(tokens=(0,)), max_states=10
    )

    assert result.status == "INCOMPLETE"
    assert result.sequence is None
    assert result.incomplete_reasons == ("MARKING_LIMIT",)
    assert result.explored_state_count == 1


def test_multiple_envelope_cuts_report_sorted_limit_names() -> None:
    # Two independent pumps: one successor crosses the token envelope while
    # further new states hit the state cap, so both limit names must appear,
    # sorted and unique.
    net = PetriNet(
        place_count=2,
        transition_count=2,
        pre=((0, 0), (0, 0)),
        post=((1, 0), (0, 1)),
    )
    result = marking_reachability(
        net, Marking(tokens=(999, 0)), Marking(tokens=(0, 0)), max_states=3
    )
    assert result.status == "INCOMPLETE"
    assert result.sequence is None
    assert result.incomplete_reasons == ("MARKING_LIMIT", "STATE_LIMIT")
    assert result.explored_state_count == 3


def test_witness_longer_than_replay_bound_is_incomplete() -> None:
    net = PetriNet.model_validate(
        {
            "place_count": 3,
            "transition_count": 2,
            "pre": [[1, 0], [0, 1], [0, 0]],
            "post": [[0, 1], [1, 0], [0, 1]],
        }
    )
    result = marking_reachability(
        net,
        Marking(tokens=(1, 0, 0)),
        Marking(tokens=(1, 0, 600)),
        max_states=2000,
    )

    assert result.status == "INCOMPLETE"
    assert result.sequence is None
    assert result.incomplete_reasons == ("SEQUENCE_LIMIT",)
    assert result.explored_state_count == 1201
