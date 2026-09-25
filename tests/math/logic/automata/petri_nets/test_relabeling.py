"""Exact place/transition axis reindexing for weighted Petri nets."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.automata.petri_nets._models import (
    PetriNetRelabelingResult,
)
from jacobian.math.logic.automata.petri_nets.operations import relabel_petri_net
from jacobian.math.logic.automata.petri_nets.values import PetriNet


def _manual_fire(
    net: PetriNet, marking: tuple[int, ...], transition: int
) -> tuple[int, ...] | None:
    """Independent definition-level enabledness and firing oracle."""
    if any(marking[p] < net.pre[p][transition] for p in range(net.place_count)):
        return None
    return tuple(
        marking[p] - net.pre[p][transition] + net.post[p][transition]
        for p in range(net.place_count)
    )


def _transport_marking(
    marking: tuple[int, ...], source_to_target: tuple[int, ...]
) -> tuple[int, ...]:
    target = [0] * len(source_to_target)
    for source, target_index in enumerate(source_to_target):
        target[target_index] = marking[source]
    return tuple(target)


def test_weighted_relabeling_preserves_arcs_ids_and_firing() -> None:
    net = PetriNet(
        place_count=2,
        transition_count=2,
        place_ids=("buffer", "output"),
        transition_ids=("load", "unload"),
        pre=((2, 0), (0, 1)),
        post=((0, 1), (1, 0)),
    )
    place_map = (1, 0)
    transition_map = (1, 0)
    result = relabel_petri_net(net, place_map, transition_map)

    assert result.target_net.place_ids == ("output", "buffer")
    assert result.target_net.transition_ids == ("unload", "load")
    assert result.target_net.pre == ((1, 0), (0, 2))
    assert result.target_net.post == ((0, 1), (1, 0))

    source_marking = (2, 0)
    mapped_marking = _transport_marking(source_marking, place_map)
    source_target = _manual_fire(net, source_marking, 0)
    mapped_target = _manual_fire(result.target_net, mapped_marking, transition_map[0])
    assert source_target == (0, 1)
    assert mapped_target == _transport_marking(source_target, place_map)

    restored = PetriNetRelabelingResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_relabeling_supports_empty_axes() -> None:
    result = relabel_petri_net(
        PetriNet(place_count=0, transition_count=0, pre=(), post=()), (), ()
    )
    assert result.target_net == result.source_net
    assert result.place_source_to_target == result.transition_source_to_target == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("place_source_to_target", (0, 0)),
        ("place_source_to_target", (0,)),
        ("transition_source_to_target", (1, 1)),
    ],
)
def test_relabeling_requires_complete_axis_bijections(
    field: str, value: tuple[int, ...]
) -> None:
    net = PetriNet(
        place_count=2,
        transition_count=2,
        pre=((0, 0), (0, 0)),
        post=((0, 0), (0, 0)),
    )
    place_map = value if field == "place_source_to_target" else (0, 1)
    transition_map = value if field == "transition_source_to_target" else (0, 1)
    with pytest.raises(OperationDomainValidationError):
        relabel_petri_net(net, place_map, transition_map)
