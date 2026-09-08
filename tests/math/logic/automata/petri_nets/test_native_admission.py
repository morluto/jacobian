"""Resource refusal remains non-completion for authored Petri claims."""

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.petri_nets._models import (
    SiphonTrapRequest,
    SiphonTrapResult,
)
from jacobian.math.logic.automata.petri_nets._tools import compute_siphon_trap
from jacobian.math.logic.automata.petri_nets.operations import verify_siphon_trap
from jacobian.math.logic.automata.petri_nets.values import PetriNet, PetriPlaceSubset


@pytest.mark.parametrize("places", [0, 1, 21, 64])
def test_zero_transition_siphon_trap_uses_singleton_closed_form(places: int) -> None:
    # With no transitions, every singleton is a minimal nonempty siphon and trap.
    net = PetriNet(
        place_count=places, transition_count=0, pre=((),) * places, post=((),) * places
    )
    subsets = tuple(PetriPlaceSubset(places=(i,)) for i in range(places))
    claim = SiphonTrapResult(net=net, siphons=subsets, traps=subsets)
    assert verify_siphon_trap(type(claim).model_validate_json(claim.model_dump_json()))
    assert compute_siphon_trap(SiphonTrapRequest(net=net)) == claim


def test_nontrivial_siphon_trap_enumeration_refusal_remains_typed() -> None:
    net = PetriNet(
        place_count=21,
        transition_count=12,
        pre=tuple(
            tuple(int((i * 7 + j * 3) % 11 < 3) for j in range(12)) for i in range(21)
        ),
        post=tuple(
            tuple(int((i * 5 + j * 2) % 13 < 4) for j in range(12)) for i in range(21)
        ),
    )
    claim = SiphonTrapResult(net=net, siphons=(), traps=())
    with pytest.raises(OperationResourceAdmissionError, match="places"):
        verify_siphon_trap(claim)
    with pytest.raises(OperationResourceAdmissionError, match="places"):
        compute_siphon_trap(SiphonTrapRequest(net=net))
