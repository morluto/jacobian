"""Exact structural profiles for a caller-selected place set."""

from __future__ import annotations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.math.logic.automata.petri_nets import (
    PetriNet,
    PetriPlaceSubset,
    place_set_support,
)


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


def test_catalog_declares_the_support_profile_operation() -> None:
    tool = Catalog.open().operation("petri_net.place_set.support_profile.compute")
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
