"""Tests for Petri-net P/T-invariants (#3762)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.petri_nets._models import (
    PetriInvariantsRequest,
    PetriInvariantsResult,
)
from jacobian.math.logic.automata.petri_nets._tools import (
    TOOLS,
    compute_petri_invariants,
)
from jacobian.math.logic.automata.petri_nets.operations import (
    petri_invariants,
    verify_invariants,
)
from jacobian.math.logic.automata.petri_nets.values import PetriNet


def _producer_consumer_net() -> PetriNet:
    """One place: t0 produces a token, t1 consumes it."""
    return PetriNet(
        place_count=1,
        transition_count=2,
        pre=((0, 1),),
        post=((1, 0),),
    )


def _marked_graph_cycle_net() -> PetriNet:
    """Two places, two transitions cycling one token."""
    return PetriNet(
        place_count=2,
        transition_count=2,
        pre=((1, 0), (0, 1)),
        post=((0, 1), (1, 0)),
    )


def _incidence_rows(net: PetriNet) -> list[list[int]]:
    return [
        [net.post[place][t] - net.pre[place][t] for t in range(net.transition_count)]
        for place in range(net.place_count)
    ]


class TestKnownAnswer:
    def test_producer_consumer_invariants(self) -> None:
        result = petri_invariants(_producer_consumer_net())

        assert isinstance(result, PetriInvariantsResult)
        assert result.incidence_rank == 1
        assert result.p_invariants == ()
        assert result.t_invariants == ((1, 1),)
        assert result.replayed

    def test_marked_graph_cycle_invariants(self) -> None:
        result = petri_invariants(_marked_graph_cycle_net())

        assert result.incidence_rank == 1
        assert result.p_invariants == ((1, 1),)
        assert result.t_invariants == ((1, 1),)

    def test_incidence_matrix_is_retained(self) -> None:
        result = petri_invariants(_producer_consumer_net())

        assert result.incidence.entries == ((1, -1),)
        assert result.incidence.row_count == 1
        assert result.incidence.column_count == 2


class TestPreservationReplay:
    @pytest.mark.parametrize(
        "net",
        [_producer_consumer_net(), _marked_graph_cycle_net()],
        ids=["producer_consumer", "marked_graph_cycle"],
    )
    def test_bases_satisfy_the_incidence_equations(self, net: PetriNet) -> None:
        result = petri_invariants(net)
        incidence = _incidence_rows(net)

        for vector in result.t_invariants:
            assert len(vector) == net.transition_count
            for place in range(net.place_count):
                assert (
                    sum(
                        incidence[place][t] * vector[t]
                        for t in range(net.transition_count)
                    )
                    == 0
                )
        for vector in result.p_invariants:
            assert len(vector) == net.place_count
            for transition in range(net.transition_count):
                assert (
                    sum(
                        vector[place] * incidence[place][transition]
                        for place in range(net.place_count)
                    )
                    == 0
                )

    @pytest.mark.parametrize(
        "net",
        [_producer_consumer_net(), _marked_graph_cycle_net()],
        ids=["producer_consumer", "marked_graph_cycle"],
    )
    def test_rank_profile_matches_the_basis_sizes(self, net: PetriNet) -> None:
        result = petri_invariants(net)

        assert len(result.t_invariants) == (
            net.transition_count - result.incidence_rank
        )
        assert len(result.p_invariants) == (net.place_count - result.incidence_rank)


class TestBoundary:
    def test_oversized_net_is_rejected(self) -> None:
        wide = PetriNet(
            place_count=17,
            transition_count=1,
            pre=tuple((0,) for _ in range(17)),
            post=tuple((0,) for _ in range(17)),
        )
        with pytest.raises(OperationResourceAdmissionError, match="at most"):
            petri_invariants(wide)

    def test_empty_place_set_gives_unit_t_invariants(self) -> None:
        net = PetriNet(place_count=0, transition_count=2, pre=(), post=())
        result = petri_invariants(net)

        assert result.incidence_rank == 0
        assert result.p_invariants == ()
        assert result.t_invariants == ((0, 1), (1, 0))

    def test_empty_transition_set_gives_unit_p_invariants(self) -> None:
        net = PetriNet(place_count=2, transition_count=0, pre=((), ()), post=((), ()))
        result = petri_invariants(net)

        assert result.incidence_rank == 0
        assert result.p_invariants == ((0, 1), (1, 0))
        assert result.t_invariants == ()


class TestAdversarial:
    def test_mismatched_pre_shape_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PetriNet.model_validate(
                {
                    "place_count": 2,
                    "transition_count": 2,
                    "pre": [[1, 0]],
                    "post": [[0, 0], [0, 0]],
                }
            )

    def test_negative_arc_weight_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PetriNet.model_validate(
                {
                    "place_count": 1,
                    "transition_count": 1,
                    "pre": [[-1]],
                    "post": [[0]],
                }
            )


class TestNativeCatalogParity:
    def test_native_kernel_and_owner_adapter_agree(self) -> None:
        request = PetriInvariantsRequest(net=_marked_graph_cycle_net())

        assert compute_petri_invariants(request) == petri_invariants(request.net)

    def test_declared_example_executes_through_the_owner_adapter(self) -> None:
        operation = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "petri_net.invariants.compute"
        )
        request = operation.request_type.model_validate_json(
            encode_strict_json(operation.examples[0].input), strict=True
        )
        result = operation.run(request)

        assert result.incidence_rank == 1
        assert result.t_invariants == ((1, 1),)


class TestSerialization:
    def test_strict_json_round_trip_preserves_the_bases(self) -> None:
        result = compute_petri_invariants(
            PetriInvariantsRequest(net=_marked_graph_cycle_net())
        )
        decoded = PetriInvariantsResult.model_validate_json(result.model_dump_json())

        assert decoded == result
        assert verify_invariants(decoded)

    def test_forged_invariants_do_not_verify(self) -> None:
        result = compute_petri_invariants(
            PetriInvariantsRequest(net=_marked_graph_cycle_net())
        )
        forged = result.model_copy(update={"t_invariants": ((1, 0),)})

        assert not verify_invariants(forged)
