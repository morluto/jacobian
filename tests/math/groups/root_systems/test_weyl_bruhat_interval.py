"""Exact complete Bruhat intervals for small finite Weyl groups."""

from __future__ import annotations

import itertools

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.root_systems._models import (
    CartanMatrix,
    WeylBruhatIntervalResult,
    WeylElement,
)
from jacobian.math.groups.root_systems.operations import (
    weyl_bruhat_interval,
    weyl_element_from_word,
    weyl_longest_element,
)

A2 = ((2, -1), (-1, 2))
A3 = ((2, -1, 0), (-1, 2, -1), (0, -1, 2))


def _permutation(action: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Recover the S_n permutation from its exact action on simple roots."""
    rank = len(action)
    result = [-1] * (rank + 1)
    for simple_index in range(rank):
        simple_coordinates = tuple(action[row][simple_index] for row in range(rank))
        ambient = (
            simple_coordinates[0],
            *(
                simple_coordinates[index] - simple_coordinates[index - 1]
                for index in range(1, rank)
            ),
            -simple_coordinates[-1],
        )
        positive = [index for index, value in enumerate(ambient) if value == 1]
        negative = [index for index, value in enumerate(ambient) if value == -1]
        assert len(positive) == len(negative) == 1
        result[simple_index] = positive[0]
        result[simple_index + 1] = negative[0]
    assert sorted(result) == list(range(rank + 1))
    return tuple(result)


def _permutation_bruhat_leq(lower: tuple[int, ...], upper: tuple[int, ...]) -> bool:
    """Independent type-A rank-matrix criterion for strong Bruhat order."""
    size = len(lower)
    return all(
        sum(value <= threshold for value in lower[:prefix])
        >= sum(value <= threshold for value in upper[:prefix])
        for prefix in range(1, size + 1)
        for threshold in range(size)
    )


def test_a3_full_interval_matches_independent_s4_rank_matrix_oracle() -> None:
    parent = CartanMatrix.model_validate(A3)
    identity = weyl_element_from_word(parent, ())
    longest = weyl_element_from_word(parent, weyl_longest_element(parent).word)

    result = weyl_bruhat_interval(identity, longest)
    actual = {
        _permutation(element.root_action.entries): label
        for element, label in zip(result.elements, result.poset.elements, strict=True)
    }
    permutations = tuple(itertools.permutations(range(4)))
    assert set(actual) == set(permutations)
    expected_strict = {
        (actual[lower], actual[upper])
        for lower in permutations
        for upper in permutations
        if lower != upper and _permutation_bruhat_leq(lower, upper)
    }
    assert {
        (pair.lower, pair.upper) for pair in result.poset.strict_order_pairs
    } == expected_strict
    assert result.poset.ranks is not None
    rank_for = {entry.element: entry.rank for entry in result.poset.ranks}
    for permutation, label in actual.items():
        assert rank_for[label] == sum(
            permutation[i] > permutation[j] for i in range(4) for j in range(i + 1, 4)
        )

    covers = {(pair.lower, pair.upper) for pair in result.poset.cover_relations}
    reachable = set(covers)
    while True:
        expanded = reachable | {
            (lower, upper)
            for lower, middle in reachable
            for next_middle, upper in reachable
            if middle == next_middle
        }
        if expanded == reachable:
            break
        reachable = expanded
    assert reachable == expected_strict


def test_empty_equal_and_nonempty_intervals_round_trip_with_endpoint_binding() -> None:
    parent = CartanMatrix.model_validate(A2)
    identity = weyl_element_from_word(parent, ())
    s0 = weyl_element_from_word(parent, (0,))
    s1 = weyl_element_from_word(parent, (1,))

    empty = weyl_bruhat_interval(s0, s1)
    assert empty.elements == ()
    assert empty.poset.elements == ()
    assert empty.lower == s0 and empty.upper == s1

    singleton = weyl_bruhat_interval(s0, s0)
    assert len(singleton.elements) == 1
    assert singleton.elements[0] == s0
    assert singleton.poset.minimal_elements == singleton.poset.maximal_elements

    chain = weyl_bruhat_interval(identity, s0)
    revived = WeylBruhatIntervalResult.model_validate_json(chain.model_dump_json())
    assert revived == chain
    label_for = {
        element.root_action.entries: label
        for element, label in zip(chain.elements, chain.poset.elements, strict=True)
    }
    assert {(pair.lower, pair.upper) for pair in chain.poset.strict_order_pairs} == {
        (label_for[identity.root_action.entries], label_for[s0.root_action.entries])
    }


def test_parent_mismatch_invalid_weyl_action_and_group_admission_are_rejected() -> None:
    a2_identity = weyl_element_from_word(A2, ())
    b2_identity = weyl_element_from_word(((2, -2), (-1, 2)), ())
    with pytest.raises(OperationDomainValidationError, match="same ordered Cartan"):
        weyl_bruhat_interval(a2_identity, b2_identity)

    invalid = WeylElement.model_construct(
        matrix=a2_identity.matrix,
        root_action=a2_identity.root_action.model_copy(
            update={"entries": ((2, 0), (0, 1))}
        ),
    )
    with pytest.raises(OperationDomainValidationError):
        weyl_bruhat_interval(invalid, a2_identity)

    a4 = ((2, -1, 0, 0), (-1, 2, -1, 0), (0, -1, 2, -1), (0, 0, -1, 2))
    identity = weyl_element_from_word(a4, ())
    with pytest.raises(OperationResourceAdmissionError, match="order at most 64"):
        weyl_bruhat_interval(identity, identity)


def test_catalog_example_executes_the_declared_operation() -> None:
    from jacobian.canonical import encode_strict_json
    from jacobian.math.groups.root_systems._tools import TOOLS

    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "weyl_group.bruhat_interval.compute"
    )
    assert tool.examples
    payload = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    assert tool.run(payload) == weyl_bruhat_interval(payload.lower, payload.upper)
