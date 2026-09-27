"""Exact structural profiles for a caller-selected place set."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.automata.petri_nets import (
    PetriNet,
    PetriPlaceSubset,
    place_set_support,
)
from jacobian.math.logic.automata.petri_nets._models import PlaceSetSupportResult
from jacobian.math.logic.automata.petri_nets._tools import TOOLS


def test_profile_returns_complete_support_and_both_predicates() -> None:
    # t0 produces into places 0,1 and consumes from 1; t1 consumes from 0
    # but produces outside the selected set; t2 produces into the set only.
    net = PetriNet(
        place_count=3,
        transition_count=3,
        pre=((0, 1, 0), (1, 0, 0), (0, 0, 1)),
        post=((1, 0, 1), (1, 0, 0), (0, 1, 1)),
    )
    result = place_set_support(net, PetriPlaceSubset(places=(0, 1)))

    assert result.producers_into == (0, 2)
    assert result.consumers_from == (0, 1)
    assert result.siphon_offenders == (2,)
    assert result.trap_offenders == (1,)
    assert not result.is_siphon
    assert not result.is_trap
    assert result.model_validate_json(result.model_dump_json()) == result


def test_empty_place_set_has_vacuous_support_predicates() -> None:
    net = PetriNet(place_count=0, transition_count=2, pre=(), post=())
    result = place_set_support(net, PetriPlaceSubset())
    assert result.producers_into == result.consumers_from == ()
    assert result.siphon_offenders == result.trap_offenders == ()
    assert result.is_siphon and result.is_trap


def test_profile_rejects_a_place_outside_its_net_axis() -> None:
    net = PetriNet(place_count=1, transition_count=0, pre=((),), post=((),))
    with pytest.raises(ValueError, match="subset must use the net place axis"):
        place_set_support(net, PetriPlaceSubset(places=(1,)))


@pytest.mark.parametrize("malformed", [None, [0], (0,), {"places": (0,)}, 1])
def test_native_malformed_place_sets_get_owner_domain_errors(malformed) -> None:
    net = PetriNet(
        place_count=2, transition_count=1, pre=((1,), (0,)), post=((0,), (1,))
    )
    with pytest.raises(OperationDomainValidationError):
        place_set_support(net, malformed)


def test_native_out_of_axis_place_gets_the_stable_domain_error() -> None:
    net = PetriNet(
        place_count=2, transition_count=1, pre=((1,), (0,)), post=((0,), (1,))
    )
    with pytest.raises(OperationDomainValidationError) as excinfo:
        place_set_support(net, PetriPlaceSubset(places=(2,)))
    assert type(excinfo.value) is OperationDomainValidationError
    assert excinfo.value.errors()[0]["type"] == "petri_net.place_axis"


def test_support_profiles_match_exhaustive_arc_enumeration() -> None:
    net = PetriNet(
        place_count=3,
        transition_count=3,
        pre=((1, 0, 1), (0, 1, 0), (0, 0, 0)),
        post=((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    )
    for mask in range(1 << 3):
        places = tuple(p for p in range(3) if mask >> p & 1)
        result = place_set_support(net, PetriPlaceSubset(places=places))
        selected = set(places)
        producers = sorted(
            {t for t in range(3) for p in selected if net.post[p][t] > 0}
        )
        consumers = sorted({t for t in range(3) for p in selected if net.pre[p][t] > 0})
        assert result.producers_into == tuple(producers)
        assert result.consumers_from == tuple(consumers)
        assert result.siphon_offenders == tuple(
            t for t in producers if t not in consumers
        )
        assert result.trap_offenders == tuple(
            t for t in consumers if t not in producers
        )
        assert result.is_siphon == (not result.siphon_offenders)
        assert result.is_trap == (not result.trap_offenders)


def test_serialized_profile_rejects_negative_transition_indices() -> None:
    # Shared -1 entries keep the offender-set equalities consistent, so only
    # the transition-axis check can reject the forged wire profile.
    net = PetriNet(place_count=1, transition_count=2, pre=((0, 0),), post=((0, 0),))
    result = place_set_support(net, PetriPlaceSubset(places=(0,)))
    payload = json.loads(result.model_dump_json())
    payload["producers_into"] = [-1]
    payload["consumers_from"] = [-1]
    with pytest.raises(ValidationError):
        PlaceSetSupportResult.model_validate_json(json.dumps(payload))


def test_catalog_declares_the_support_profile_operation() -> None:
    tool = next(
        t
        for t in TOOLS
        if t.operation_id == "petri_net.place_set.support_profile.compute"
    )
    assert tool is not None
    result = tool.run(
        tool.request_type.model_validate(
            {
                "net": {
                    "place_count": 1,
                    "transition_count": 1,
                    "pre": [[1]],
                    "post": [[1]],
                },
                "places": {"places": [0]},
            }
        )
    )
    assert result.is_siphon and result.is_trap
    assert result.producers_into == result.consumers_from == (0,)
