"""Complete siphon/trap enumeration agrees with the support definition."""

from itertools import combinations
from random import Random

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.petri_nets._models import (
    SiphonTrapFamilyRequest,
)
from jacobian.math.logic.automata.petri_nets._tools import (
    compute_siphon_trap_family,
)
from jacobian.math.logic.automata.petri_nets.operations import siphon_trap_family
from jacobian.math.logic.automata.petri_nets.values import PetriNet


def _brute_families(net: PetriNet) -> tuple[tuple[tuple[int, ...], ...], ...]:
    subsets = [
        subset
        for size in range(1, net.place_count + 1)
        for subset in combinations(range(net.place_count), size)
    ]

    def member(subset: tuple[int, ...], *, siphon: bool) -> bool:
        incoming = net.post if siphon else net.pre
        outgoing = net.pre if siphon else net.post
        for transition in range(net.transition_count):
            produces = any(incoming[place][transition] > 0 for place in subset)
            consumes = any(outgoing[place][transition] > 0 for place in subset)
            if produces and not consumes:
                return False
        return True

    return tuple(
        tuple(subset for subset in subsets if member(subset, siphon=kind))
        for kind in (True, False)
    )


def test_complete_families_match_independent_brute_force_support_oracle() -> None:
    random = Random(1908)
    for places in range(1, 8):
        for _ in range(10):
            transitions = random.randrange(6)
            net = PetriNet(
                place_count=places,
                transition_count=transitions,
                pre=tuple(
                    tuple(random.randrange(3) for _ in range(transitions))
                    for _ in range(places)
                ),
                post=tuple(
                    tuple(random.randrange(3) for _ in range(transitions))
                    for _ in range(places)
                ),
            )
            result = siphon_trap_family(net)
            assert (
                tuple(item.places for item in result.siphons) == _brute_families(net)[0]
            )
            assert (
                tuple(item.places for item in result.traps) == _brute_families(net)[1]
            )


def test_tool_family_is_bound_to_source_net_and_serializes() -> None:
    net = PetriNet(
        place_count=2,
        transition_count=2,
        pre=((1, 0), (0, 1)),
        post=((0, 1), (1, 0)),
    )
    result = compute_siphon_trap_family(SiphonTrapFamilyRequest(net=net))
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert tuple(item.places for item in decoded.siphons) == ((0, 1),)
    assert decoded.siphons == decoded.traps
    assert decoded.net == net


def test_full_family_output_is_preflight_bounded() -> None:
    places = 16
    net = PetriNet(
        place_count=places,
        transition_count=0,
        pre=tuple(() for _ in range(places)),
        post=tuple(() for _ in range(places)),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        siphon_trap_family(net)
    assert (
        error.value.errors()[0]["type"] == "petri_net.siphon_trap_family_output_bound"
    )
