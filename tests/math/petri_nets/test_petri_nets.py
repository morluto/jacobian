"""Tests for Petri net operations."""

import pytest
from pydantic import ValidationError

from jacobian.math import petri_nets
from jacobian.math.petri_nets._models import (
    EnabledTransitionsRequest,
    FireTransitionRequest,
    IncidenceMatrixRequest,
    ReachabilityRequest,
    ReachabilityResult,
)
from jacobian.math.petri_nets._operations import (
    compute_enabled_transitions,
    compute_fire_transition,
    compute_incidence,
    compute_reachability,
)
from jacobian.math.petri_nets._tools import TOOLS
from jacobian.math.petri_nets.values import (
    MAX_PETRI_ARC_WEIGHT,
    MAX_PETRI_MARKING,
    MAX_REACHABILITY_EXPLORATION_WORK,
    MAX_REACHABILITY_FIRING_RECORDS,
    MAX_REACHABILITY_STATE_TOKEN_CELLS,
    Marking,
    PetriNet,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_net() -> PetriNet:
    """2 places, 2 transitions: t0 moves a token from p0 to p1."""
    return PetriNet(
        place_count=2,
        transition_count=2,
        pre=((1, 0), (0, 1)),
        post=((0, 0), (0, 1)),
    )


def _token_passing_net() -> PetriNet:
    """Net where t0: p0->p1 and t1: p1->p0 (cyclic)."""
    return PetriNet(
        place_count=2,
        transition_count=2,
        pre=((1, 0), (0, 1)),
        post=((0, 1), (1, 0)),
    )


def test_catalog_contains_only_audited_agent_outcomes() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "petri_net.fire_transition.compute",
        "petri_net.reachability_graph.compute",
    }


def test_exploratory_operations_remain_native() -> None:
    net = _token_passing_net()
    marking = petri_nets.Marking(tokens=(1, 0))
    assert petri_nets.enabled_transitions(net, marking) == [0]
    assert petri_nets.compute_incidence_matrix(net) == ((-1, 1), (1, -1))


def test_native_kernel_validates_cross_field_inputs() -> None:
    with pytest.raises(ValueError, match="marking length"):
        petri_nets.enabled_transitions(_simple_net(), Marking(tokens=(1,)))
    with pytest.raises(ValueError, match="transition index"):
        petri_nets.fire_transition(_simple_net(), Marking(tokens=(1, 0)), 2)
    with pytest.raises(ValueError, match="max_states"):
        petri_nets.reachability_graph(
            _simple_net(), Marking(tokens=(1, 0)), max_states=0
        )


# ---------------------------------------------------------------------------
# Enabled transitions
# ---------------------------------------------------------------------------


class TestEnabledTransitions:
    def test_simple_enabled(self):
        net = _simple_net()
        marking = Marking(tokens=(2, 0))
        result = compute_enabled_transitions(
            EnabledTransitionsRequest(net=net, marking=marking)
        )
        assert result.transitions == (0,)

    def test_none_enabled(self):
        net = _simple_net()
        marking = Marking(tokens=(0, 0))
        result = compute_enabled_transitions(
            EnabledTransitionsRequest(net=net, marking=marking)
        )
        assert result.transitions == ()

    def test_both_enabled(self):
        net = _token_passing_net()
        marking = Marking(tokens=(1, 1))
        result = compute_enabled_transitions(
            EnabledTransitionsRequest(net=net, marking=marking)
        )
        assert result.transitions == (0, 1)


# ---------------------------------------------------------------------------
# Fire transition
# ---------------------------------------------------------------------------


class TestFireTransition:
    def test_fire_success(self):
        net = _simple_net()
        marking = Marking(tokens=(2, 0))
        result = compute_fire_transition(
            FireTransitionRequest(net=net, marking=marking, transition=0)
        )
        assert result.status == "FIRED"
        assert result.new_marking == Marking(tokens=(1, 0))
        assert result.envelope_escape is None

    def test_fire_disabled(self):
        net = _simple_net()
        marking = Marking(tokens=(0, 0))
        result = compute_fire_transition(
            FireTransitionRequest(net=net, marking=marking, transition=0)
        )
        assert result.status == "NOT_ENABLED"
        assert result.new_marking == Marking(tokens=(0, 0))
        assert result.envelope_escape is None

    def test_fire_cyclic(self):
        net = _token_passing_net()
        marking = Marking(tokens=(1, 0))
        result = compute_fire_transition(
            FireTransitionRequest(net=net, marking=marking, transition=0)
        )
        assert result.status == "FIRED"
        assert result.new_marking == Marking(tokens=(0, 1))
        assert result.envelope_escape is None

    def test_fire_reports_successor_outside_marking_envelope(self):
        net = PetriNet(
            place_count=1,
            transition_count=1,
            pre=((0,),),
            post=((MAX_PETRI_ARC_WEIGHT,),),
        )
        result = compute_fire_transition(
            FireTransitionRequest(
                net=net,
                marking=Marking(tokens=(MAX_PETRI_MARKING,)),
                transition=0,
            )
        )

        assert result.status == "ESCAPES_DECLARED_ENVELOPE"
        assert result.new_marking is None
        assert result.envelope_escape == (2 * MAX_PETRI_MARKING,)


# ---------------------------------------------------------------------------
# Incidence matrix
# ---------------------------------------------------------------------------


class TestIncidenceMatrix:
    def test_simple_incidence(self):
        net = _simple_net()
        result = compute_incidence(IncidenceMatrixRequest(net=net))
        assert result.incidence == ((-1, 0), (0, 0))

    def test_cyclic_incidence(self):
        net = _token_passing_net()
        result = compute_incidence(IncidenceMatrixRequest(net=net))
        assert result.incidence == ((-1, 1), (1, -1))


# ---------------------------------------------------------------------------
# Reachability graph
# ---------------------------------------------------------------------------


class TestReachability:
    def test_simple_reachability(self):
        net = _simple_net()
        marking = Marking(tokens=(2, 0))
        result = compute_reachability(
            ReachabilityRequest(net=net, initial_marking=marking, max_states=100)
        )
        # From (2,0): fire t0 -> (1,0), fire t0 again -> (0,0)
        assert (2, 0) in result.states
        assert result.status == "COMPLETE"
        assert result.frontier == ()

    def test_cyclic_reachability(self):
        net = _token_passing_net()
        marking = Marking(tokens=(1, 0))
        result = compute_reachability(
            ReachabilityRequest(net=net, initial_marking=marking, max_states=100)
        )
        # Cyclic: (1,0) -> t0 -> (0,1) -> t1 -> (1,0)
        assert len(result.states) == 2
        assert (1, 0) in result.states
        assert (0, 1) in result.states
        assert result.status == "COMPLETE"
        assert result.frontier == ()

    def test_truncation(self):
        net = _token_passing_net()
        marking = Marking(tokens=(1, 0))
        result = compute_reachability(
            ReachabilityRequest(net=net, initial_marking=marking, max_states=1)
        )
        assert result.status == "TRUNCATED"
        assert result.states == ((1, 0),)
        assert tuple(
            (record.source_state, record.transition, record.target_marking)
            for record in result.frontier
        ) == ((0, 0, (0, 1)),)

    def test_exact_state_limit_is_complete(self):
        result = compute_reachability(
            ReachabilityRequest(
                net=_token_passing_net(),
                initial_marking=Marking(tokens=(1, 0)),
                max_states=2,
            )
        )
        assert result.status == "COMPLETE"
        assert result.frontier == ()

    def test_max_states_one_self_loop_is_complete(self):
        net = PetriNet(
            place_count=1,
            transition_count=1,
            pre=((1,),),
            post=((1,),),
        )
        result = compute_reachability(
            ReachabilityRequest(
                net=net,
                initial_marking=Marking(tokens=(1,)),
                max_states=1,
            )
        )
        assert result.status == "COMPLETE"
        assert result.states == ((1,),)
        assert result.edges == ((0, 0, 0),)
        assert result.frontier == ()

    def test_unbounded_net_exposes_replayable_frontier(self):
        net = PetriNet(
            place_count=1,
            transition_count=1,
            pre=((0,),),
            post=((1,),),
        )
        result = compute_reachability(
            ReachabilityRequest(
                net=net,
                initial_marking=Marking(tokens=(0,)),
                max_states=3,
            )
        )
        assert result.status == "TRUNCATED"
        assert result.states == ((0,), (1,), (2,))
        assert result.edges == ((0, 0, 1), (1, 0, 2))
        assert len(result.frontier) == 1
        assert result.frontier[0].target_marking == (3,)

    def test_marking_growth_returns_replayable_envelope_escape(self):
        net = PetriNet(
            place_count=1,
            transition_count=1,
            pre=((0,),),
            post=((1,),),
        )
        result = compute_reachability(
            ReachabilityRequest(
                net=net,
                initial_marking=Marking(tokens=(MAX_PETRI_MARKING,)),
                max_states=10,
            )
        )
        assert result.status == "ESCAPES_DECLARED_ENVELOPE"
        assert result.states == ((MAX_PETRI_MARKING,),)
        assert result.frontier == ()
        assert result.envelope_escape is not None
        assert result.envelope_escape.source_state == 0
        assert result.envelope_escape.transition == 0
        assert result.envelope_escape.target_marking == (MAX_PETRI_MARKING + 1,)

    def test_result_rejects_false_complete_status(self):
        result = compute_reachability(
            ReachabilityRequest(
                net=_token_passing_net(),
                initial_marking=Marking(tokens=(1, 0)),
                max_states=1,
            )
        )
        payload = result.model_dump(mode="json")
        payload["status"] = "COMPLETE"
        with pytest.raises(ValidationError, match="deterministic BFS outcome"):
            ReachabilityResult.model_validate(payload)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_wrong_pre_dimensions_rejected(self):
        with pytest.raises(ValidationError):
            PetriNet(
                place_count=2,
                transition_count=2,
                pre=((1, 0),),
                post=((0, 0), (0, 0)),
            )

    def test_negative_marking_rejected(self):
        with pytest.raises(ValidationError):
            Marking(tokens=(-1, 0))

    def test_negative_arc_weight_rejected(self):
        with pytest.raises(ValidationError):
            PetriNet(
                place_count=2,
                transition_count=1,
                pre=((-1, 0), (0, 0)),
                post=((0, 0), (0, 0)),
            )

    @pytest.mark.parametrize("value", [MAX_PETRI_ARC_WEIGHT, MAX_PETRI_MARKING])
    def test_scalar_bounds_accept_boundary(self, value: int):
        PetriNet(
            place_count=1,
            transition_count=1,
            pre=((value,),),
            post=((value,),),
        )
        Marking(tokens=(value,))

    def test_scalar_bounds_reject_above_boundary(self):
        with pytest.raises(ValidationError, match="pre weights"):
            PetriNet(
                place_count=1,
                transition_count=1,
                pre=((MAX_PETRI_ARC_WEIGHT + 1,),),
                post=((0,),),
            )
        with pytest.raises(ValidationError, match="marking tokens"):
            Marking(tokens=(MAX_PETRI_MARKING + 1,))

    def test_reachability_bounds_state_cells(self):
        place_count = 64
        net = PetriNet(
            place_count=place_count,
            transition_count=1,
            pre=tuple((0,) for _ in range(place_count)),
            post=tuple((0,) for _ in range(place_count)),
        )
        accepted_states = MAX_REACHABILITY_STATE_TOKEN_CELLS // place_count
        ReachabilityRequest(
            net=net,
            initial_marking=Marking(tokens=(0,) * place_count),
            max_states=accepted_states,
        )
        with pytest.raises(ValidationError, match="state-token cells"):
            ReachabilityRequest(
                net=net,
                initial_marking=Marking(tokens=(0,) * place_count),
                max_states=accepted_states + 1,
            )

    def test_reachability_bounds_firing_records(self):
        transition_count = 64
        net = PetriNet(
            place_count=1,
            transition_count=transition_count,
            pre=((0,) * transition_count,),
            post=((0,) * transition_count,),
        )
        accepted_states = MAX_REACHABILITY_FIRING_RECORDS // transition_count
        ReachabilityRequest(
            net=net,
            initial_marking=Marking(tokens=(0,)),
            max_states=accepted_states,
        )
        with pytest.raises(ValidationError, match="firing records"):
            ReachabilityRequest(
                net=net,
                initial_marking=Marking(tokens=(0,)),
                max_states=accepted_states + 1,
            )

    def test_reachability_bounds_exploration_work(self):
        dimension = 64
        net = PetriNet(
            place_count=dimension,
            transition_count=dimension,
            pre=tuple((0,) * dimension for _ in range(dimension)),
            post=tuple((0,) * dimension for _ in range(dimension)),
        )
        accepted_states = MAX_REACHABILITY_EXPLORATION_WORK // (2 * dimension**2)
        ReachabilityRequest(
            net=net,
            initial_marking=Marking(tokens=(0,) * dimension),
            max_states=accepted_states,
        )
        with pytest.raises(ValidationError, match="exploration"):
            ReachabilityRequest(
                net=net,
                initial_marking=Marking(tokens=(0,) * dimension),
                max_states=accepted_states + 1,
            )

    def test_reachability_bounds_estimated_result_bytes(self):
        net = PetriNet(
            place_count=2,
            transition_count=2,
            pre=((0, 0), (0, 0)),
            post=((0, 0), (0, 0)),
        )
        marking = Marking(tokens=(0, 0))
        ReachabilityRequest(net=net, initial_marking=marking, max_states=48_390)
        with pytest.raises(ValidationError, match="byte bound"):
            ReachabilityRequest(
                net=net,
                initial_marking=marking,
                max_states=48_391,
            )

    def test_native_reachability_uses_the_same_aggregate_admission(self):
        dimension = 64
        net = PetriNet(
            place_count=dimension,
            transition_count=dimension,
            pre=tuple((0,) * dimension for _ in range(dimension)),
            post=tuple((0,) * dimension for _ in range(dimension)),
        )
        with pytest.raises(ValueError, match="reachability"):
            petri_nets.reachability_graph(
                net,
                Marking(tokens=(0,) * dimension),
                max_states=100_000,
            )

    def test_transition_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            FireTransitionRequest(
                net=_simple_net(),
                marking=Marking(tokens=(1, 0)),
                transition=5,
            )
