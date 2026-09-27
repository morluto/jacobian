"""Exact sequential commutation profiles for Petri transition pairs."""

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.petri_nets import Marking, PetriNet
from jacobian.math.logic.automata.petri_nets._models import (
    MarkingCommutationProfileRequest,
    MarkingCommutationProfileResult,
)
from jacobian.math.logic.automata.petri_nets._tools import TOOLS
from jacobian.math.logic.automata.petri_nets.operations import (
    marking_commutation_profile,
)


def _independent_replay(net: PetriNet, tokens: tuple[int, ...], order: tuple[int, int]):
    """Small direct oracle using only the P/T firing definition."""

    current = list(tokens)
    prefixes: list[tuple[int, ...]] = []
    for step, transition in enumerate(order):
        deficit = tuple(
            max(0, net.pre[place][transition] - current[place])
            for place in range(net.place_count)
        )
        if any(deficit):
            return ("BLOCKED", tuple(prefixes), None, step, deficit)
        current = [
            current[place] - net.pre[place][transition] + net.post[place][transition]
            for place in range(net.place_count)
        ]
        prefixes.append(tuple(current))
    return ("FIRES", tuple(prefixes), tuple(current), None, None)


def _observed_replay(replay):
    return (
        replay.status,
        tuple(marking.tokens for marking in replay.prefix_markings),
        None if replay.final_marking is None else replay.final_marking.tokens,
        replay.blocked_index,
        replay.deficit,
    )


def test_both_orders_match_independent_firing_oracle_exhaustively() -> None:
    # All one-place two-transition nets with binary arc weights, at token
    # counts 0, 1, and 2. This includes blocked, one-order-only, and diamond
    # outcomes without using Jacobian's replay operation as the oracle.
    for arcs in product(range(2), repeat=4):
        net = PetriNet(
            place_count=1,
            transition_count=2,
            pre=((arcs[0], arcs[1]),),
            post=((arcs[2], arcs[3]),),
        )
        for tokens in ((0,), (1,), (2,)):
            result = marking_commutation_profile(net, Marking(tokens=tokens), (0, 1))
            forward = _independent_replay(net, tokens, (0, 1))
            backward = _independent_replay(net, tokens, (1, 0))
            assert _observed_replay(result.first_then_second) == forward
            assert _observed_replay(result.second_then_first) == backward
            both_fire = forward[0] == backward[0] == "FIRES"
            same_target = bool(both_fire and forward[2] == backward[2])
            assert result.both_orders_fire is both_fire
            assert result.same_target is same_target


def test_independent_self_loops_form_a_sequential_diamond() -> None:
    net = PetriNet(
        place_count=2,
        transition_count=2,
        pre=((1, 0), (0, 1)),
        post=((1, 0), (0, 1)),
    )
    result = marking_commutation_profile(net, Marking(tokens=(1, 1)), (0, 1))
    assert result.both_orders_fire
    assert result.same_target
    assert (
        result.first_then_second.final_marking == result.second_then_first.final_marking
    )
    assert tuple(item.tokens for item in result.first_then_second.prefix_markings) == (
        (1, 1),
        (1, 1),
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_only_the_order_that_produces_first_fires() -> None:
    net = PetriNet(
        place_count=1,
        transition_count=2,
        pre=((1, 0),),
        post=((0, 1),),
    )
    result = marking_commutation_profile(net, Marking(tokens=(0,)), (0, 1))
    assert result.first_then_second.status == "BLOCKED"
    assert result.first_then_second.blocked_index == 0
    assert result.first_then_second.deficit == (1,)
    assert result.second_then_first.status == "FIRES"
    assert result.second_then_first.final_marking is not None
    assert result.second_then_first.final_marking.tokens == (0,)
    assert not result.both_orders_fire
    assert not result.same_target


def test_request_and_native_api_reject_invalid_transition_pairs() -> None:
    net = PetriNet(
        place_count=0,
        transition_count=2,
        pre=(),
        post=(),
    )
    with pytest.raises(ValidationError, match="distinct transitions"):
        MarkingCommutationProfileRequest(
            net=net, marking=Marking(tokens=()), transitions=(0, 0)
        )
    with pytest.raises(OperationDomainValidationError, match="distinct transitions"):
        marking_commutation_profile(net, Marking(tokens=()), (1, 1))


def test_result_rejects_authored_flags_that_disagree_with_replays() -> None:
    net = PetriNet(
        place_count=0,
        transition_count=2,
        pre=(),
        post=(),
    )
    result = marking_commutation_profile(net, Marking(tokens=()), (0, 1))
    forged = result.model_dump(mode="json")
    forged["same_target"] = False
    with pytest.raises(ValidationError, match="flags must agree"):
        MarkingCommutationProfileResult.model_validate(forged)


def test_catalog_publishes_the_sequential_profile_separately_from_conflicts() -> None:
    tool = next(
        t
        for t in TOOLS
        if t.operation_id == "petri_net.marking.commutation_profile.compute"
    )
    assert tool is not None
    request = tool.request_type.model_validate(
        {
            "net": {
                "place_count": 1,
                "transition_count": 2,
                "pre": [[1, 0]],
                "post": [[0, 1]],
            },
            "marking": {"tokens": [0]},
            "transitions": [0, 1],
        }
    )
    result = tool.run(request)
    assert isinstance(result, MarkingCommutationProfileResult)
    assert result.first_then_second.status == "BLOCKED"
    assert result.second_then_first.status == "FIRES"
    catalog_example = tool.request_type.model_validate(tool.examples[0].input)
    assert tool.run(catalog_example) == result
    # This is a local sequential-order result, not the simultaneous profile.
    conflict = next(
        t
        for t in TOOLS
        if t.operation_id == "petri_net.marking.conflict_profile.compute"
    )
    assert conflict is not None
    assert "sequential commutation" in conflict.description
    assert "global" in tool.description


def test_output_is_admitted_before_two_replays_are_built() -> None:
    net = PetriNet(
        place_count=1,
        transition_count=2,
        place_ids=("p" * (2 * 1024 * 1024),),
        pre=((0, 0),),
        post=((0, 0),),
    )
    marking = Marking(tokens=(0,), net=net)
    with pytest.raises(
        OperationResourceAdmissionError, match="serialized output bound"
    ):
        marking_commutation_profile(net, marking, (0, 1))
