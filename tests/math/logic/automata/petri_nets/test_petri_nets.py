"""Tests for Petri net operations."""

# These behavior-focused test methods follow pytest's unannotated method
# convention; Petri production modules are type-checked.
# mypy: disable-error-code=no-untyped-def

import json

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata import petri_nets
from jacobian.math.logic.automata.petri_nets._models import (
    MAX_SIPHON_TRAP_PLACES,
    EnabledTransitionsRequest,
    FireTransitionRequest,
    FireTransitionResult,
    FiringSequenceReplayResult,
    IncidenceMatrixRequest,
    ReachabilityRequest,
    SiphonTrapRequest,
    StateEquationRequest,
)
from jacobian.math.logic.automata.petri_nets._tools import (
    compute_enabled_transitions,
    compute_fire_transition,
    compute_incidence,
    compute_reachability,
    compute_siphon_trap,
)
from jacobian.math.logic.automata.petri_nets.values import Marking, PetriNet

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


@pytest.mark.parametrize(
    "operation",
    [
        petri_nets.compute_incidence_matrix,
        petri_nets.find_minimal_siphons,
        petri_nets.find_minimal_traps,
        petri_nets.petri_invariants,
    ],
)
def test_native_operations_reject_model_constructed_net(operation) -> None:
    forged = PetriNet.model_construct(
        place_count=1, transition_count=1, pre=None, post=None
    )
    with pytest.raises(OperationDomainValidationError):
        operation(forged)


def test_native_operations_return_canonical_results() -> None:
    net = _simple_net()
    marking = Marking(tokens=(1, 0))

    assert petri_nets.enabled_transitions(net, marking).transitions == (0,)
    assert petri_nets.fire_transition(net, marking, 0).status == "FIRED"
    assert petri_nets.compute_incidence_matrix(net).incidence.entries == (
        (-1, 0),
        (0, 0),
    )
    assert petri_nets.reachability_graph(net, marking).states[0].marking.tokens == (
        1,
        0,
    )


def test_state_equation_is_formal_and_does_not_claim_reachability() -> None:
    # Each transition consumes a token from the place produced by the other.
    # At the empty marking neither can fire, but y=(1,1) satisfies C y = 0.
    net = PetriNet(
        place_count=2,
        transition_count=2,
        pre=((1, 0), (0, 1)),
        post=((0, 1), (1, 0)),
    )
    source = Marking(tokens=(0, 0), net=net)
    request = StateEquationRequest(net=net, marking=source, transition_counts=(1, 1))

    result = petri_nets.state_equation_target(
        request.net, request.marking, request.transition_counts
    )
    assert result.target == (0, 0)
    assert result.marking.net == net
    assert petri_nets.enabled_transitions(net, source).transitions == ()
    assert result.model_validate_json(result.model_dump_json()) == result


def test_state_equation_returns_signed_integer_target() -> None:
    net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((1,),),
        post=((0,),),
    )
    result = petri_nets.state_equation_target(net, Marking(tokens=(0,)), (1,))
    assert result.target == (-1,)


@pytest.mark.parametrize(
    "forged_target",
    [
        [64_001_001],
        [0] * 65,
    ],
)
def test_state_equation_result_rejects_out_of_envelope_serialized_targets(
    forged_target,
) -> None:
    net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((0,),),
        post=((0,),),
    )
    valid = petri_nets.state_equation_target(net, Marking(tokens=(0,)), (0,))
    payload = valid.model_dump()
    payload["target"] = forged_target
    with pytest.raises(ValidationError):
        type(valid).model_validate_json(json.dumps(payload))


def test_state_equation_rejects_foreign_marking_and_unbounded_count_vector() -> None:
    net = _simple_net()
    foreign = _token_passing_net()
    with pytest.raises(ValidationError):
        StateEquationRequest(
            net=net,
            marking=Marking(tokens=(1, 0), net=foreign),
            transition_counts=(0, 0),
        )
    parsed = StateEquationRequest(
        net=net,
        marking=Marking(tokens=(1, 0)),
        transition_counts=(1001, 0),
    )
    with pytest.raises(OperationResourceAdmissionError):
        petri_nets.state_equation_target(
            parsed.net, parsed.marking, parsed.transition_counts
        )


@pytest.mark.parametrize(
    "malformed",
    [None, [0, 0], "ab", 1, {0: 0, 1: 0}, (0, 0, 0), (0.0, 0), (0, -1), (True, 0)],
)
def test_state_equation_rejects_malformed_count_containers(malformed) -> None:
    # The exported native face admits the same canonical tuple as the wire
    # model before taking the vector's length, so malformed containers get
    # the stable domain error instead of a TypeError or lax coercion.
    net = _simple_net()
    with pytest.raises(OperationDomainValidationError):
        petri_nets.state_equation_target(net, Marking(tokens=(1, 0)), malformed)


def test_state_equation_canonical_vector_still_matches_direct_arithmetic() -> None:
    net = _token_passing_net()
    tokens = (2, 0)
    result = petri_nets.state_equation_target(
        net, Marking(tokens=tokens, net=net), (1, 0)
    )
    expected = tuple(
        tokens[p] + (net.post[p][0] - net.pre[p][0]) for p in range(net.place_count)
    )
    assert result.target == expected
    assert result.transition_counts == (1, 0)


def test_empty_net_preserves_empty_axes_across_json() -> None:
    net = PetriNet(place_count=0, transition_count=0, pre=(), post=())
    marking = Marking(tokens=())

    enabled = petri_nets.enabled_transitions(net, marking)
    incidence = petri_nets.compute_incidence_matrix(net)
    reachability = petri_nets.reachability_graph(net, marking, max_states=1)
    siphons = petri_nets.find_minimal_siphons(net)
    traps = petri_nets.find_minimal_traps(net)

    assert enabled.transitions == ()
    assert incidence.incidence.row_count == 0
    assert incidence.incidence.column_count == 0
    assert incidence.incidence.entries == ()
    assert reachability.states[0].place_axis == ()
    assert reachability.states[0].marking.tokens == ()
    assert reachability.edges == ()
    assert siphons == []
    assert traps == []
    decoded = type(reachability).model_validate_json(reachability.model_dump_json())
    assert decoded == reachability
    assert petri_nets.verify_reachability_graph(decoded)


def test_every_result_marking_retains_the_declared_net_parent() -> None:
    net = _token_passing_net()
    foreign = PetriNet(
        place_count=2,
        transition_count=2,
        pre=((0, 1), (1, 0)),
        post=((1, 0), (0, 1)),
    )
    foreign_marking = Marking(tokens=(0, 1), net=foreign)
    with pytest.raises(ValidationError):
        FireTransitionResult(
            net=net,
            marking=Marking(tokens=(1, 0), net=net),
            transition=0,
            status="FIRED",
            new_marking=foreign_marking,
        )
    replay = petri_nets.replay_firing_sequence(
        net, Marking(tokens=(1, 0), net=net), (0,)
    )
    payload = replay.model_dump(mode="json")
    payload["prefix_markings"][0] = foreign_marking.model_dump(mode="json")
    with pytest.raises(ValidationError):
        FiringSequenceReplayResult.model_validate_json(json.dumps(payload))
    reachability = petri_nets.reachability_graph(
        net, Marking(tokens=(1, 0), net=net), max_states=10
    )
    payload = reachability.model_dump(mode="json")
    payload["states"][0]["marking"] = foreign_marking.model_dump(mode="json")
    with pytest.raises(ValidationError):
        type(reachability).model_validate_json(json.dumps(payload))


def test_typed_petri_state_axes_reject_forged_serialized_claim() -> None:
    net = _token_passing_net()
    result = petri_nets.reachability_graph(net, Marking(tokens=(1, 0)), max_states=10)
    decoded = type(result).model_validate_json(result.model_dump_json())
    forged_state = decoded.states[0].model_copy(update={"place_axis": (1, 0)})
    forged = decoded.model_copy(update={"states": (forged_state, *decoded.states[1:])})
    assert not petri_nets.verify_reachability_graph(forged)


def test_native_operations_reject_mismatched_markings() -> None:
    net = _simple_net()
    marking = Marking(tokens=(1,))

    with pytest.raises(ValueError, match="marking length"):
        petri_nets.enabled_transitions(net, marking)
    with pytest.raises(ValueError, match="marking length"):
        petri_nets.fire_transition(net, marking, 0)
    with pytest.raises(ValueError, match="marking length"):
        petri_nets.reachability_graph(net, marking)


def test_serialized_claims_retain_sources_and_reject_forgery() -> None:
    net = _token_passing_net()
    marking = Marking(tokens=(1, 0))
    enabled = compute_enabled_transitions(
        EnabledTransitionsRequest(net=net, marking=marking)
    )
    enabled_decoded = type(enabled).model_validate_json(enabled.model_dump_json())
    assert enabled_decoded.net == net
    assert enabled_decoded.marking == marking
    assert petri_nets.verify_enabled_transitions(enabled_decoded)
    assert not petri_nets.verify_enabled_transitions(
        enabled_decoded.model_copy(update={"transitions": (1,)})
    )

    fired = compute_fire_transition(
        FireTransitionRequest(net=net, marking=marking, transition=0)
    )
    fired_decoded = type(fired).model_validate_json(fired.model_dump_json())
    assert petri_nets.verify_fire_transition(fired_decoded)
    assert not petri_nets.verify_fire_transition(
        fired_decoded.model_copy(update={"status": "NOT_ENABLED"})
    )

    incidence = compute_incidence(IncidenceMatrixRequest(net=net))
    incidence_decoded = type(incidence).model_validate_json(incidence.model_dump_json())
    assert petri_nets.verify_incidence_matrix(incidence_decoded)
    assert not petri_nets.verify_incidence_matrix(
        incidence_decoded.model_copy(update={"incidence": ((0, 0), (0, 0))})
    )

    reachability = compute_reachability(
        ReachabilityRequest(net=net, initial_marking=marking, max_states=10)
    )
    reachability_decoded = type(reachability).model_validate_json(
        reachability.model_dump_json()
    )
    assert reachability_decoded.initial_marking == marking
    assert petri_nets.verify_reachability_graph(reachability_decoded)
    assert not petri_nets.verify_reachability_graph(
        reachability_decoded.model_copy(update={"truncated": True})
    )

    siphon = compute_siphon_trap(SiphonTrapRequest(net=net))
    siphon_decoded = type(siphon).model_validate_json(siphon.model_dump_json())
    assert siphon_decoded.net == net
    assert petri_nets.verify_siphon_trap(siphon_decoded)
    assert not petri_nets.verify_siphon_trap(
        siphon_decoded.model_copy(update={"siphons": ()})
    )
    forged_payload = siphon.model_dump(mode="json")
    forged_payload["siphons"] = [{"places": [-1]}]
    with pytest.raises(ValidationError):
        type(siphon).model_validate(forged_payload)


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
        assert result.incidence.entries == ((-1, 0), (0, 0))

    def test_cyclic_incidence(self):
        net = _token_passing_net()
        result = compute_incidence(IncidenceMatrixRequest(net=net))
        assert result.incidence.entries == ((-1, 1), (1, -1))


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
        assert (2, 0) in tuple(state.marking.tokens for state in result.states)
        assert not result.truncated

    def test_cyclic_reachability(self):
        net = _token_passing_net()
        marking = Marking(tokens=(1, 0))
        result = compute_reachability(
            ReachabilityRequest(net=net, initial_marking=marking, max_states=100)
        )
        # Cyclic: (1,0) -> t0 -> (0,1) -> t1 -> (1,0)
        assert len(result.states) == 2
        assert (1, 0) in tuple(state.marking.tokens for state in result.states)
        assert (0, 1) in tuple(state.marking.tokens for state in result.states)
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

from jacobian.math.logic.automata.petri_nets.operations import (  # noqa: E402
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
            assert all(0 <= p < 2 for p in s.places)
        for t in result.traps:
            assert all(0 <= p < 2 for p in t.places)

    def test_siphon_trap_sorted(self):
        """Siphons and traps should be sorted tuples."""
        net = _cyclic_net()
        result = compute_siphon_trap(SiphonTrapRequest(net=net))
        for s in result.siphons:
            assert list(s.places) == sorted(s.places)
        for t in result.traps:
            assert list(t.places) == sorted(t.places)

    def test_siphon_trap_one_way(self):
        """One-way net: {0} is a siphon (not a trap), {1} is a trap."""
        net = _one_way_net()
        result = compute_siphon_trap(SiphonTrapRequest(net=net))
        siphon_sets = [frozenset(s.places) for s in result.siphons]
        trap_sets = [frozenset(t.places) for t in result.traps]
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


def test_siphon_trap_ignores_transitions_with_empty_incidence() -> None:
    net = PetriNet(
        place_count=MAX_SIPHON_TRAP_PLACES,
        transition_count=64,
        pre=tuple((0,) * 64 for _ in range(MAX_SIPHON_TRAP_PLACES)),
        post=tuple((0,) * 64 for _ in range(MAX_SIPHON_TRAP_PLACES)),
    )
    request = SiphonTrapRequest(net=net)
    result = compute_siphon_trap(request)
    assert tuple(item.places for item in result.siphons) == tuple(
        (i,) for i in range(net.place_count)
    )
    assert result.siphons == result.traps
