"""Exact local pairwise simultaneous-step compatibility profiles."""

from itertools import combinations, product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.petri_nets import Marking, PetriNet
from jacobian.math.logic.automata.petri_nets._models import (
    MAX_MARKING_CONFLICT_PROFILE_PAIRS,
    MarkingConflictProfileResult,
)
from jacobian.math.logic.automata.petri_nets.operations import marking_conflict_profile


def test_pairwise_profile_matches_direct_aggregate_enabling_oracle():
    # Exhaust small weighted nets and markings; the oracle checks each
    # transition alone and each distinct pair from the defining Pre inequality.
    for pre_entries in product(range(2), repeat=4):
        net = PetriNet(
            place_count=2,
            transition_count=2,
            pre=(pre_entries[:2], pre_entries[2:]),
            post=((0, 0), (0, 0)),
        )
        for tokens in product(range(2), repeat=2):
            marking = Marking(tokens=tokens)
            individually_enabled = tuple(
                transition
                for transition in range(2)
                if all(tokens[p] >= net.pre[p][transition] for p in range(2))
            )
            result = marking_conflict_profile(net, marking)
            assert result.enabled_transitions == individually_enabled
            expected = tuple(combinations(individually_enabled, 2))
            jointly_enabled = tuple(
                pair
                for pair in expected
                if all(
                    tokens[p] >= net.pre[p][pair[0]] + net.pre[p][pair[1]]
                    for p in range(2)
                )
            )
            assert result.jointly_enabled_pairs == jointly_enabled
            assert result.conflicting_pairs == tuple(
                pair for pair in expected if pair not in jointly_enabled
            )


def test_profile_is_source_bound_and_round_trips():
    net = PetriNet(
        place_count=1,
        transition_count=3,
        pre=((1, 1, 0),),
        post=((0, 0, 1),),
    )
    marking = Marking(tokens=(1,), net=net)
    result = marking_conflict_profile(net, marking)
    assert result.enabled_transitions == (0, 1, 2)
    assert result.jointly_enabled_pairs == ((0, 2), (1, 2))
    assert result.conflicting_pairs == ((0, 1),)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_forged_pair_families_are_rejected_before_pair_partition_work():
    net = PetriNet(
        place_count=0,
        transition_count=64,
        pre=(),
        post=(),
    )
    oversized = {
        "net": net.model_dump(),
        "marking": {"tokens": []},
        "enabled_transitions": list(range(64)),
        # Each field is individually at its schema maximum, but their total
        # exceeds the 64-choose-2 partition. Validation must reject based on
        # cardinality before concatenating and deduplicating both families.
        "jointly_enabled_pairs": [[0, 1]] * MAX_MARKING_CONFLICT_PROFILE_PAIRS,
        "conflicting_pairs": [[0, 1]],
    }
    with pytest.raises(ValidationError, match="cardinality exceeds enabled pairs"):
        MarkingConflictProfileResult.model_validate(oversized)

    field_oversized = {
        **oversized,
        "jointly_enabled_pairs": [[0, 1]] * (MAX_MARKING_CONFLICT_PROFILE_PAIRS + 1),
        "conflicting_pairs": [],
    }
    with pytest.raises(ValidationError, match="at most"):
        MarkingConflictProfileResult.model_validate(field_oversized)


def test_profile_result_size_is_admitted_before_pair_materialization():
    net = PetriNet(
        place_count=1,
        transition_count=1,
        place_ids=("p" * (10 * 1024 * 1024),),
        pre=((0,),),
        post=((0,),),
    )
    with pytest.raises(OperationResourceAdmissionError, match="serialized result"):
        marking_conflict_profile(net, Marking(tokens=(0,)))
