"""Tests for Petri net operations."""

# The behavior-focused legacy test methods intentionally follow pytest's
# unannotated method convention; Petri production modules are type-checked.
# mypy: disable-error-code=no-untyped-def

import pytest
from pydantic import ValidationError

from jacobian.math.petri_nets._models import (
    EnabledTransitionsRequest,
    FireTransitionRequest,
    IncidenceMatrixRequest,
    ReachabilityRequest,
    SiphonTrapRequest,
)
from jacobian.math.petri_nets._operations import (
    compute_enabled_transitions,
    compute_fire_transition,
    compute_incidence,
    compute_reachability,
)
from jacobian.math.petri_nets.values import Marking, PetriNet

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
        assert result.new_marking is not None
        assert result.new_marking.tokens == (1, 0)

    def test_fire_disabled(self):
        net = _simple_net()
        marking = Marking(tokens=(0, 0))
        result = compute_fire_transition(
            FireTransitionRequest(net=net, marking=marking, transition=0)
        )
        assert result.status == "NOT_ENABLED"
        assert result.new_marking is not None
        assert result.new_marking.tokens == (0, 0)

    def test_fire_cyclic(self):
        net = _token_passing_net()
        marking = Marking(tokens=(1, 0))
        result = compute_fire_transition(
            FireTransitionRequest(net=net, marking=marking, transition=0)
        )
        assert result.status == "FIRED"
        assert result.new_marking is not None
        assert result.new_marking.tokens == (0, 1)


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
        assert not result.truncated

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
        assert not result.truncated

    def test_truncation(self):
        net = _token_passing_net()
        marking = Marking(tokens=(1, 0))
        result = compute_reachability(
            ReachabilityRequest(net=net, initial_marking=marking, max_states=1)
        )
        assert result.truncated


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

    def test_transition_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            FireTransitionRequest(
                net=_simple_net(),
                marking=Marking(tokens=(1, 0)),
                transition=5,
            )


# ---------------------------------------------------------------------------
# Siphon and trap detection
# ---------------------------------------------------------------------------

from jacobian.math.petri_nets._operations import compute_siphon_trap  # noqa: E402
from jacobian.math.petri_nets.operations import (  # noqa: E402
    find_minimal_siphons,
    find_minimal_traps,
)


def _cyclic_net() -> PetriNet:
    """Net where t0: p0->p1 and t1: p1->p0 (cyclic)."""
    return PetriNet(
        place_count=2,
        transition_count=2,
        pre=((1, 0), (0, 1)),
        post=((0, 1), (1, 0)),
    )


def _one_way_net() -> PetriNet:
    """Net where t0: p0->p1 only (one-way, no cycle)."""
    return PetriNet(
        place_count=2,
        transition_count=1,
        pre=((1,), (0,)),
        post=((0,), (1,)),
    )


def _self_loop_net() -> PetriNet:
    """Net where t0: p0->p0 (self-loop)."""
    return PetriNet(
        place_count=1,
        transition_count=1,
        pre=((1,),),
        post=((1,),),
    )


class TestSiphons:
    def test_cyclic_siphons(self):
        """In a cyclic net, {0,1} is the minimal siphon."""
        net = _cyclic_net()
        siphons = find_minimal_siphons(net)
        siphon_sets = [frozenset(s) for s in siphons]
        assert frozenset({0, 1}) in siphon_sets

    def test_self_loop_siphon(self):
        """A self-loop place {0} is a siphon."""
        net = _self_loop_net()
        siphons = find_minimal_siphons(net)
        siphon_sets = [frozenset(s) for s in siphons]
        assert frozenset({0}) in siphon_sets

    def test_one_way_siphon(self):
        """In the one-way net, {0} is a siphon.

        t0 outputs to p1, not p0, so {0} satisfies the siphon condition
        vacuously (post(t0) intersection {0} = empty set).
        """
        net = _one_way_net()
        siphons = find_minimal_siphons(net)
        siphon_sets = [frozenset(s) for s in siphons]
        assert frozenset({0}) in siphon_sets


class TestTraps:
    def test_cyclic_traps(self):
        """In a cyclic net, {0,1} is the minimal trap."""
        net = _cyclic_net()
        traps = find_minimal_traps(net)
        trap_sets = [frozenset(t) for t in traps]
        assert frozenset({0, 1}) in trap_sets

    def test_self_loop_trap(self):
        """A self-loop place {0} is a trap."""
        net = _self_loop_net()
        traps = find_minimal_traps(net)
        trap_sets = [frozenset(t) for t in traps]
        assert frozenset({0}) in trap_sets

    def test_one_way_no_trap_p0(self):
        """In the one-way net, {0} is NOT a trap.

        t0 inputs from p0 and outputs to p1, so {0} is not a trap.
        {1} IS a trap (vacuously, no transition inputs from {1}).
        """
        net = _one_way_net()
        traps = find_minimal_traps(net)
        trap_sets = [frozenset(t) for t in traps]
        assert frozenset({0}) not in trap_sets
        assert frozenset({1}) in trap_sets


class TestSiphonTrapAdapter:
    def test_siphon_trap_check(self):
        net = _cyclic_net()
        result = compute_siphon_trap(SiphonTrapRequest(net=net))
        assert len(result.siphons) >= 1
        assert len(result.traps) >= 1
        for s in result.siphons:
            assert all(0 <= p < 2 for p in s)
        for t in result.traps:
            assert all(0 <= p < 2 for p in t)

    def test_siphon_trap_sorted(self):
        """Siphons and traps should be sorted tuples."""
        net = _cyclic_net()
        result = compute_siphon_trap(SiphonTrapRequest(net=net))
        for s in result.siphons:
            assert list(s) == sorted(s)
        for t in result.traps:
            assert list(t) == sorted(t)

    def test_siphon_trap_one_way(self):
        """One-way net: {0} is a siphon (not a trap), {1} is a trap."""
        net = _one_way_net()
        result = compute_siphon_trap(SiphonTrapRequest(net=net))
        siphon_sets = [frozenset(s) for s in result.siphons]
        trap_sets = [frozenset(t) for t in result.traps]
        assert frozenset({0}) in siphon_sets
        assert frozenset({0}) not in trap_sets
        assert frozenset({1}) in trap_sets


def test_petri_values_enforce_advertised_arc_and_marking_bounds() -> None:
    assert PetriNet(place_count=1, transition_count=1, pre=((1000,),), post=((1000,),))
    assert Marking(tokens=(1000,))
    with pytest.raises(ValidationError, match="pre weights must not exceed"):
        PetriNet(place_count=1, transition_count=1, pre=((1001,),), post=((0,),))
    with pytest.raises(ValidationError, match="marking tokens must not exceed"):
        Marking(tokens=(1001,))


def test_siphon_trap_admission_charges_transition_scan_work() -> None:
    net = PetriNet(
        place_count=20,
        transition_count=64,
        pre=tuple((0,) * 64 for _ in range(20)),
        post=tuple((0,) * 64 for _ in range(20)),
    )
    with pytest.raises(ValidationError, match="candidate and transition-scan work"):
        SiphonTrapRequest(net=net)
