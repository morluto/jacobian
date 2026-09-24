"""Exact transition-reversal law for finite weighted Petri nets."""

from __future__ import annotations

from itertools import product

import pytest

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.petri_nets import operations
from jacobian.math.logic.automata.petri_nets._tools import TOOLS
from jacobian.math.logic.automata.petri_nets.operations import reverse_petri_net
from jacobian.math.logic.automata.petri_nets.values import PetriNet


def _matrix(values: tuple[int, ...], places: int, transitions: int):
    return tuple(
        tuple(values[p * transitions + t] for t in range(transitions))
        for p in range(places)
    )


def test_reverse_is_involutive_and_reverses_every_tiny_weighted_firing() -> None:
    for places in (1, 2):
        for transitions in (1, 2):
            cells = places * transitions
            for arcs in product(range(3), repeat=2 * cells):
                net = PetriNet(
                    place_count=places,
                    transition_count=transitions,
                    pre=_matrix(arcs[:cells], places, transitions),
                    post=_matrix(arcs[cells:], places, transitions),
                )
                reversed_net = reverse_petri_net(net)
                assert reversed_net.pre == net.post
                assert reversed_net.post == net.pre
                assert reverse_petri_net(reversed_net) == net

                for marking in product(range(3), repeat=places):
                    for transition in range(transitions):
                        if any(
                            marking[p] < net.pre[p][transition] for p in range(places)
                        ):
                            continue
                        successor = tuple(
                            marking[p]
                            - net.pre[p][transition]
                            + net.post[p][transition]
                            for p in range(places)
                        )
                        assert all(
                            successor[p] >= reversed_net.pre[p][transition]
                            for p in range(places)
                        )
                        predecessor = tuple(
                            successor[p]
                            - reversed_net.pre[p][transition]
                            + reversed_net.post[p][transition]
                            for p in range(places)
                        )
                        assert predecessor == marking


def test_reverse_preserves_ordered_axes_and_handles_empty_net() -> None:
    net = PetriNet(
        place_count=2,
        transition_count=1,
        place_ids=("source", "target"),
        transition_ids=("move",),
        pre=((2,), (0,)),
        post=((0,), (2,)),
    )
    reversed_net = reverse_petri_net(net)
    assert reversed_net.place_ids == net.place_ids
    assert reversed_net.transition_ids == net.transition_ids
    assert reversed_net == PetriNet(
        place_count=2,
        transition_count=1,
        place_ids=("source", "target"),
        transition_ids=("move",),
        pre=((0,), (2,)),
        post=((2,), (0,)),
    )
    empty = PetriNet(place_count=0, transition_count=0, pre=(), post=())
    assert reverse_petri_net(empty) == empty


def test_reverse_is_catalogued_with_a_direct_petrit_net_value() -> None:
    tool = next(
        tool for tool in TOOLS if tool.operation_id == "petri_net.reverse.compute"
    )
    source = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((1,),),
        post=((0,),),
    )
    assert tool.request_type is PetriNet
    assert tool.result_type is PetriNet
    assert tool.run(source) == reverse_petri_net(source)


def test_reverse_output_bound_matches_canonical_json_size() -> None:
    net = PetriNet(
        place_count=2,
        transition_count=2,
        place_ids=("p\x00\n", "東京/é"),
        transition_ids=('"move\\', "𝄞"),
        pre=((0, 12), (1000, 7)),
        post=((999, 0), (1, 23)),
    )
    expected = len(encode_strict_json(net.model_dump(mode="json")))
    assert operations._petri_net_reverse_output_bound(net) == expected


def test_reverse_rejects_before_constructing_an_oversized_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((1,),),
        post=((0,),),
    )
    bound = operations._petri_net_reverse_output_bound(net)
    monkeypatch.setattr(operations, "MAX_PETRI_NET_REVERSE_OUTPUT_BYTES", bound - 1)

    def unexpected_construct(**_values: object) -> PetriNet:
        raise AssertionError("the result should be rejected before construction")

    monkeypatch.setattr(PetriNet, "model_construct", unexpected_construct)
    with pytest.raises(OperationResourceAdmissionError):
        reverse_petri_net(net)
