"""Exact checks for supplied Petri-net pumping sequences."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
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


@pytest.mark.parametrize("length", [0, 1, 2, 3, 5, 64])
def test_growth_witnesses_match_independent_replay_oracle(length: int) -> None:
    # t0 recycles its p0 token and grows p1, so any repetition count is a
    # genuine pumping witness with exact delta (0, length).
    net = PetriNet(
        place_count=2, transition_count=1, pre=((1,), (0,)), post=((1,), (1,))
    )
    result = check_pumping_witness(net, Marking(tokens=(1, 0)), (0,) * length)
    assert result.status == ("PUMPING_WITNESS" if length else "FIRES_WITHOUT_GROWTH")
    assert result.growth == ((0, length) if length else None)
    assert result.replay.final_marking is not None
    assert result.replay.final_marking.tokens == (1, length)
    assert len(result.replay.prefix_markings) == length


def test_pumping_replay_ledger_is_admitted_before_prefixes_exist() -> None:
    long_id = "place-" + "x" * 40_000
    net = PetriNet(
        place_count=1,
        transition_count=1,
        place_ids=(long_id,),
        pre=((0,),),
        post=((0,),),
    )
    source = Marking(tokens=(0,), net=net)
    # Every retained prefix serializes the parent-bound source marking,
    # repeating the whole named net up to the 1024-entry admitted length.
    with pytest.raises(OperationResourceAdmissionError, match="output bound"):
        check_pumping_witness(net, source, (0,) * 1024)
    # A small ledger of the same named net stays exact.
    small = check_pumping_witness(net, source, (0,) * 10)
    assert small.status == "FIRES_WITHOUT_GROWTH"
    assert len(small.replay.prefix_markings) == 10
    # An unparented marking does not embed the net in each prefix, so the
    # full admitted length stays serializable.
    loose = check_pumping_witness(net, Marking(tokens=(0,)), (0,) * 1024)
    assert len(loose.replay.prefix_markings) == 1024


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
