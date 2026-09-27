"""Exact simultaneous-step semantics, independently checked by direct arithmetic."""

from itertools import product

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.petri_nets import (
    concurrent_step,
    replay_firing_sequence,
)
from jacobian.math.logic.automata.petri_nets._tools import TOOLS
from jacobian.math.logic.automata.petri_nets.values import Marking, PetriNet


def _oracle(net: PetriNet, marking: tuple[int, ...], counts: tuple[int, ...]):
    required = tuple(
        sum(net.pre[p][t] * counts[t] for t in range(net.transition_count))
        for p in range(net.place_count)
    )
    deficit = tuple(max(0, required[p] - marking[p]) for p in range(net.place_count))
    if any(deficit):
        return "NOT_ENABLED", required, deficit, None
    target = tuple(
        marking[p]
        - sum(net.pre[p][t] * counts[t] for t in range(net.transition_count))
        + sum(net.post[p][t] * counts[t] for t in range(net.transition_count))
        for p in range(net.place_count)
    )
    return "FIRED", required, deficit, target


def test_two_enabled_transitions_can_conflict_as_a_simultaneous_step():
    net = PetriNet(
        place_count=1,
        transition_count=2,
        pre=((1, 1),),
        post=((0, 0),),
    )
    marking = Marking(tokens=(1,))
    result = concurrent_step(net, marking, (1, 1))
    assert result.status == "NOT_ENABLED"
    assert result.required == (2,)
    assert result.deficit == (1,)


def test_step_semantics_differs_from_a_fireable_sequential_sequence():
    # t0 produces the resource consumed by t1. The simultaneous multiset asks
    # for that resource at the source marking, while the sequence may interleave.
    net = PetriNet(
        place_count=2,
        transition_count=2,
        pre=((1, 0), (0, 1)),
        post=((0, 1), (1, 0)),
    )
    marking = Marking(tokens=(1, 0))
    simultaneous = concurrent_step(net, marking, (1, 1))
    sequential = replay_firing_sequence(net, marking, (0, 1))
    assert simultaneous.status == "NOT_ENABLED"
    assert simultaneous.deficit == (0, 1)
    assert sequential.status == "FIRES"
    assert sequential.final_marking.tokens == (1, 0)


def test_small_weighted_steps_match_independent_state_equation_oracle():
    # Exhaust all 1-place/2-transition nets with arc weights 0 or 1, all
    # markings in {0,1,2}, and all transition multiplicities in {0,1,2}.
    for pre_flat in product(range(2), repeat=2):
        for post_flat in product(range(2), repeat=2):
            net = PetriNet(
                place_count=1,
                transition_count=2,
                pre=(pre_flat,),
                post=(post_flat,),
            )
            for tokens in range(3):
                for counts in product(range(3), repeat=2):
                    expected = _oracle(net, (tokens,), counts)
                    actual = concurrent_step(net, Marking(tokens=(tokens,)), counts)
                    assert (actual.status, actual.required, actual.deficit) == expected[
                        :3
                    ]
                    if expected[0] == "FIRED":
                        assert actual.new_marking is not None
                        assert actual.new_marking.tokens == expected[3]
                    else:
                        assert actual.new_marking is None


def test_declared_envelope_escape_is_not_reported_as_a_marking():
    net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((0,),),
        post=((1000,),),
    )
    result = concurrent_step(net, Marking(tokens=(1,)), (1,))
    assert result.status == "ESCAPES_DECLARED_ENVELOPE"
    assert result.envelope_escape == (1001,)
    assert result.new_marking is None


def test_catalog_invocation_has_typed_deficit_result():
    tool = next(
        t
        for t in TOOLS
        if t.operation_id == "petri_net.marking.concurrent_step.compute"
    )
    result = tool.run(
        tool.request_type.model_validate(
            {
                "net": {
                    "place_count": 1,
                    "transition_count": 2,
                    "pre": [[1, 1]],
                    "post": [[0, 0]],
                },
                "marking": {"tokens": [1]},
                "transition_counts": [1, 1],
            }
        )
    )
    assert result.status == "NOT_ENABLED"
    assert result.deficit == (1,)


def test_occurrence_cap_is_enforced_before_step_expansion():
    net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((0,),),
        post=((0,),),
    )
    with pytest.raises(OperationResourceAdmissionError, match="multiplicity"):
        concurrent_step(net, Marking(tokens=(0,)), (1001,))


@pytest.mark.parametrize(
    "malformed",
    [None, [1, 1], "ab", 1, {0: 1, 1: 1}, (1, 1, 1), (1.0, 1), (1, -1), (True, 0)],
)
def test_malformed_count_containers_get_the_stable_domain_error(malformed):
    # A direct native caller must never see a bare len() TypeError or a
    # silently coerced non-canonical container; the declared canonical
    # tuple is admitted before its length is inspected.
    net = PetriNet(place_count=1, transition_count=2, pre=((1, 1),), post=((0, 0),))
    with pytest.raises(OperationDomainValidationError):
        concurrent_step(net, Marking(tokens=(1,)), malformed)


def test_canonical_step_still_matches_the_independent_oracle():
    net = PetriNet(place_count=1, transition_count=2, pre=((1, 1),), post=((2, 2),))
    actual = concurrent_step(net, Marking(tokens=(1,)), (1, 1))
    expected = _oracle(net, (1,), (1, 1))
    assert (actual.status, actual.required, actual.deficit) == expected[:3]
    assert actual.transition_counts == (1, 1)
