"""Exact checks for supplied Petri-net pumping sequences."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.math.logic.automata.petri_nets._models import PumpingWitnessRequest
from jacobian.math.logic.automata.petri_nets.operations import check_pumping_witness
from jacobian.math.logic.automata.petri_nets.values import Marking, PetriNet


def test_repeatable_growth_sequence_is_a_real_unboundedness_witness() -> None:
    net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((0,),),
        post=((1,),),
    )
    source = Marking(tokens=(0,))

    result = check_pumping_witness(net, source, (0,))

    assert result.status == "PUMPING_WITNESS"
    assert result.growth == (1,)
    assert result.replay.final_marking is not None
    assert result.replay.final_marking.tokens == (1,)
    # Independent arithmetic oracle: after k repetitions the marking is k.
    marking = 0
    for _ in range(25):
        marking += 1
    assert marking == source.tokens[0] + 25 * result.growth[0]


@pytest.mark.parametrize(
    ("pre", "post", "source", "sequence", "status"),
    [
        (((1,),), ((1,),), (1,), (0,), "FIRES_WITHOUT_GROWTH"),
        (((1,),), ((0,),), (1,), (0,), "FIRES_WITHOUT_GROWTH"),
        (((1,),), ((2,),), (0,), (0,), "BLOCKED"),
    ],
)
def test_non_pumping_and_blocked_sequences_make_no_growth_claim(
    pre: tuple[tuple[int, ...], ...],
    post: tuple[tuple[int, ...], ...],
    source: tuple[int, ...],
    sequence: tuple[int, ...],
    status: str,
) -> None:
    net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=pre,
        post=post,
    )

    result = check_pumping_witness(net, Marking(tokens=source), sequence)

    assert result.status == status
    assert result.growth is None
    assert (result.replay.status == "BLOCKED") == (status == "BLOCKED")


def test_catalog_contract_and_source_parent_are_preserved() -> None:
    tool = Catalog.open().operation("petri_net.firing_sequence.pumping_witness.check")
    result = tool.run(
        PumpingWitnessRequest(
            net=PetriNet(
                place_count=1,
                transition_count=1,
                pre=((0,),),
                post=((1,),),
            ),
            marking=Marking(tokens=(0,)),
            sequence=(0,),
        )
    )

    assert result.status == "PUMPING_WITNESS"
    assert result.replay.net == result.net
    assert result.replay.marking == result.marking
