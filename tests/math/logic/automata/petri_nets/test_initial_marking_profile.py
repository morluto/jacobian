"""Initial-marking profiles for selected Petri-net siphons and traps."""

from __future__ import annotations

import json
from itertools import product

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.math.logic.automata.petri_nets import (
    Marking,
    PetriNet,
    PetriPlaceSubset,
    place_set_initial_marking_profile,
)


def _manual_support(net: PetriNet, places: tuple[int, ...]) -> tuple[bool, bool]:
    selected = set(places)
    producers = {
        t for t in range(net.transition_count) if any(net.post[p][t] for p in selected)
    }
    consumers = {
        t for t in range(net.transition_count) if any(net.pre[p][t] for p in selected)
    }
    return producers <= consumers, consumers <= producers


def _reachable_prefixes(net: PetriNet, initial: tuple[int, ...], depth: int):
    frontier = {initial}
    yield initial
    for _ in range(depth):
        following: set[tuple[int, ...]] = set()
        for marking in frontier:
            for transition in range(net.transition_count):
                if any(
                    marking[place] < net.pre[place][transition]
                    for place in range(net.place_count)
                ):
                    continue
                target = tuple(
                    marking[place]
                    - net.pre[place][transition]
                    + net.post[place][transition]
                    for place in range(net.place_count)
                )
                following.add(target)
        yield from following
        frontier = following


def test_small_net_exhaustion_checks_each_reported_persistence_implication() -> None:
    # Exhaust every one-transition P/T net with two places and binary arc
    # weights, every binary initial marking, and every selected place subset.
    for arcs in product((0, 1), repeat=4):
        pre = ((arcs[0],), (arcs[1],))
        post = ((arcs[2],), (arcs[3],))
        net = PetriNet(place_count=2, transition_count=1, pre=pre, post=post)
        for tokens in product((0, 1), repeat=2):
            marking = Marking(tokens=tokens, net=net)
            for mask in range(4):
                places = tuple(place for place in range(2) if mask & (1 << place))
                subset = PetriPlaceSubset(places=places)
                result = place_set_initial_marking_profile(net, subset, marking)
                total = sum(tokens[place] for place in places)
                is_siphon, is_trap = _manual_support(net, places)

                assert result.selected_token_total == total
                assert result.support_profile.is_siphon is is_siphon
                assert result.support_profile.is_trap is is_trap
                expected = tuple(
                    implication
                    for applies, implication in (
                        (is_siphon and total == 0, "EMPTY_SIPHON_REMAINS_EMPTY"),
                        (is_trap and total > 0, "MARKED_TRAP_REMAINS_MARKED"),
                    )
                    if applies
                )
                assert result.preservation_implications == expected

                for reached in _reachable_prefixes(net, tokens, depth=4):
                    reached_total = sum(reached[place] for place in places)
                    if "EMPTY_SIPHON_REMAINS_EMPTY" in expected:
                        assert reached_total == 0
                    if "MARKED_TRAP_REMAINS_MARKED" in expected:
                        assert reached_total > 0


def test_profile_retains_source_and_round_trips() -> None:
    net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((1,),),
        post=((1,),),
    )
    subset = PetriPlaceSubset(places=(0,))
    marking = Marking(tokens=(3,), net=net)
    result = place_set_initial_marking_profile(net, subset, marking)

    assert result.net == net
    assert result.places == subset
    assert result.marking == marking
    assert result.selected_token_total == 3
    assert result.preservation_implications == ("MARKED_TRAP_REMAINS_MARKED",)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_profile_rejects_mismatched_parent_and_place_axis() -> None:
    net = PetriNet(place_count=1, transition_count=0, pre=((),), post=((),))
    other = PetriNet(place_count=1, transition_count=1, pre=((0,),), post=((0,),))
    with pytest.raises(ValueError, match="different Petri net place axis"):
        place_set_initial_marking_profile(
            net, PetriPlaceSubset(), Marking(tokens=(0,), net=other)
        )
    with pytest.raises(ValueError, match="subset must use the net place axis"):
        place_set_initial_marking_profile(
            net, PetriPlaceSubset(places=(1,)), Marking(tokens=(0,))
        )


def test_catalog_publishes_the_marking_profile_operation() -> None:
    tool = Catalog.open().operation(
        "petri_net.place_set.initial_marking_profile.compute"
    )
    assert tool is not None
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert result.selected_token_total == 1
    assert result.preservation_implications == ("MARKED_TRAP_REMAINS_MARKED",)
