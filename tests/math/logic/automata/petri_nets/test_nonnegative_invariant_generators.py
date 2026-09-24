"""Exact nonnegative P/T-invariant Hilbert bases."""

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.petri_nets import (
    PetriNet,
    petri_nonnegative_invariant_generators,
)
from jacobian.math.logic.automata.petri_nets._models import (
    PetriNonnegativeInvariantResult,
)


def test_nonunimodular_transition_cone_includes_interior_generator() -> None:
    # C = [2, 3, -5]. The extreme primitive rays are (5,0,2) and
    # (0,5,3), but their integer monoid misses (1,1,1), also required.
    net = PetriNet(
        place_count=1,
        transition_count=3,
        pre=((0, 0, 5),),
        post=((2, 3, 0),),
    )
    result = petri_nonnegative_invariant_generators(net)

    assert result.net == net
    assert result.t_generators == ((0, 5, 3), (1, 1, 1), (5, 0, 2))
    assert result.p_generators == ()
    assert all(2 * x + 3 * y - 5 * z == 0 for x, y, z in result.t_generators)


def test_conservative_cycle_and_zero_incidence_axes() -> None:
    cycle = PetriNet(
        place_count=2,
        transition_count=2,
        pre=((1, 0), (0, 1)),
        post=((0, 1), (1, 0)),
    )
    result = petri_nonnegative_invariant_generators(cycle)
    assert result.p_generators == ((1, 1),)
    assert result.t_generators == ((1, 1),)

    isolated = PetriNet(
        place_count=2,
        transition_count=0,
        pre=((), ()),
        post=((), ()),
    )
    result = petri_nonnegative_invariant_generators(isolated)
    assert result.p_generators == ((1, 0), (0, 1))
    assert result.t_generators == ()


def test_result_deserializes_canonical_axis_bound_generators() -> None:
    net = PetriNet(
        place_count=1,
        transition_count=3,
        pre=((0, 0, 5),),
        post=((2, 3, 0),),
    )
    result = petri_nonnegative_invariant_generators(net)
    assert PetriNonnegativeInvariantResult.model_validate(result.model_dump()) == result
    with pytest.raises(ValueError, match="sorted and unique"):
        PetriNonnegativeInvariantResult(
            net=net,
            p_generators=(),
            t_generators=((1, 1, 1), (0, 5, 3)),
        )


def test_large_candidate_space_is_rejected_before_enumeration() -> None:
    net = PetriNet(
        place_count=1,
        transition_count=4,
        pre=((0, 0, 0, 1000),),
        post=((1000, 999, 998, 0),),
    )
    with pytest.raises(OperationResourceAdmissionError, match="Hilbert-basis"):
        petri_nonnegative_invariant_generators(net)
