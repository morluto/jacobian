"""Tests for Petri-net firing-sequence replay (#1908)."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.petri_nets._models import (
    FiringSequenceReplayRequest,
    FiringSequenceReplayResult,
)
from jacobian.math.logic.automata.petri_nets._tools import (
    compute_firing_sequence_replay,
)
from jacobian.math.logic.automata.petri_nets.operations import (
    replay_firing_sequence,
)
from jacobian.math.logic.automata.petri_nets.values import Marking, PetriNet

# Cyclic two-place net: t0 moves p0 -> p1, t1 moves p1 -> p0.
NET = PetriNet(
    place_count=2,
    transition_count=2,
    pre=((1, 0), (0, 1)),
    post=((0, 1), (1, 0)),
)


def _request(tokens: tuple[int, ...], sequence: tuple[int, ...]):
    return FiringSequenceReplayRequest(
        net=NET, marking=Marking(tokens=tokens), sequence=sequence
    )


class TestKnownAnswer:
    def test_fires_cycle(self) -> None:
        result = replay_firing_sequence(NET, Marking(tokens=(1, 0)), (0, 1))
        assert isinstance(result, FiringSequenceReplayResult)
        assert result.status == "FIRES"
        assert [m.tokens for m in result.prefix_markings] == [(0, 1), (1, 0)]
        assert result.final_marking is not None
        assert result.final_marking.tokens == (1, 0)
        assert result.parikh == (1, 1)
        assert result.state_equation_residual == (0, 0)

    def test_blocked_first_transition(self) -> None:
        result = replay_firing_sequence(NET, Marking(tokens=(1, 0)), (1, 0))
        assert result.status == "BLOCKED"
        assert result.blocked_index == 0
        assert result.prefix_markings == ()
        assert result.parikh == (0, 0)
        assert result.deficit == (0, 1)
        assert result.first_deficient_place == 1

    def test_blocked_later_transition(self) -> None:
        result = replay_firing_sequence(NET, Marking(tokens=(1, 0)), (0, 0))
        assert result.status == "BLOCKED"
        assert result.blocked_index == 1
        assert [m.tokens for m in result.prefix_markings] == [(0, 1)]
        assert result.parikh == (1, 0)
        assert result.deficit == (1, 0)
        assert result.first_deficient_place == 0

    def test_repeated_firing_accumulates(self) -> None:
        result = replay_firing_sequence(NET, Marking(tokens=(2, 0)), (0, 0))
        assert result.status == "FIRES"
        assert [m.tokens for m in result.prefix_markings] == [(1, 1), (0, 2)]
        assert result.parikh == (2, 0)


class TestBoundaryDegenerate:
    def test_empty_sequence_fires_trivially(self) -> None:
        result = replay_firing_sequence(NET, Marking(tokens=(1, 0)), ())
        assert result.status == "FIRES"
        assert result.prefix_markings == ()
        assert result.final_marking is not None
        assert result.final_marking.tokens == (1, 0)
        assert result.parikh == (0, 0)
        assert result.state_equation_residual == (0, 0)


class TestAdversarial:
    def test_transition_index_out_of_range(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            replay_firing_sequence(NET, Marking(tokens=(1, 0)), (0, 7))

    def test_marking_axis_mismatch(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            replay_firing_sequence(NET, Marking(tokens=(1, 0, 0)), (0,))

    def test_catalog_path_rejects_bad_index(self) -> None:
        request = FiringSequenceReplayRequest.model_construct(
            net=NET, marking=Marking(tokens=(1, 0)), sequence=(0, 5)
        )
        with pytest.raises(OperationDomainValidationError):
            compute_firing_sequence_replay(request)

    def test_parent_bound_ledger_is_admitted_before_prefix_construction(
        self,
    ) -> None:
        long_id = "p" + "x" * 40_000
        net = PetriNet(
            place_count=1,
            transition_count=1,
            place_ids=(long_id,),
            pre=((0,),),
            post=((0,),),
        )
        bound = Marking(tokens=(0,), net=net)
        with pytest.raises(OperationResourceAdmissionError, match="output bound"):
            replay_firing_sequence(net, bound, (0,) * 1024)
        result = replay_firing_sequence(net, bound, (0, 0))
        assert result.status == "FIRES"
        assert len(result.prefix_markings) == 2


class TestDefiningInvariant:
    @pytest.mark.parametrize(
        ("tokens", "sequence"),
        [
            ((1, 0), (0, 1)),
            ((3, 0), (0, 1, 0, 1)),
            ((0, 2), (1, 1)),
            ((2, 0), (0, 0)),
            ((1, 0), (1, 0)),
        ],
    )
    def test_state_equation_holds(
        self, tokens: tuple[int, ...], sequence: tuple[int, ...]
    ) -> None:
        result = replay_firing_sequence(NET, Marking(tokens=tokens), sequence)
        fired = len(sequence) if result.status == "FIRES" else result.blocked_index
        assert fired is not None
        assert len(result.prefix_markings) == fired
        assert sum(result.parikh) == fired
        reached = (
            result.final_marking.tokens
            if result.final_marking is not None
            else result.prefix_markings[-1].tokens
            if result.prefix_markings
            else tokens
        )
        for place in range(NET.place_count):
            assert (
                reached[place]
                - tokens[place]
                - sum(
                    (NET.post[place][t] - NET.pre[place][t]) * result.parikh[t]
                    for t in range(NET.transition_count)
                )
                == result.state_equation_residual[place]
            )
        if result.status == "FIRES":
            assert result.state_equation_residual == (0, 0)
        else:
            assert result.deficit is not None and any(
                entry > 0 for entry in result.deficit
            )


class TestNativeCatalogParity:
    @pytest.mark.parametrize(
        ("tokens", "sequence"),
        [((1, 0), (0, 1)), ((1, 0), (1, 0)), ((2, 0), ())],
    )
    def test_native_matches_catalog(
        self, tokens: tuple[int, ...], sequence: tuple[int, ...]
    ) -> None:
        request = _request(tokens, sequence)
        assert compute_firing_sequence_replay(request) == replay_firing_sequence(
            request.net, request.marking, request.sequence
        )
