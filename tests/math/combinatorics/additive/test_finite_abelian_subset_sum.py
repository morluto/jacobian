"""Exact finite abelian indexed subset-sum profiles."""

from itertools import product

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.additive.finite_abelian_subset_sum import (
    FiniteAbelianSubsetSumRequest,
    FiniteAbelianSubsetSumResult,
    finite_abelian_subset_sum_profile,
)
from jacobian.math.groups.finite_abelian import (
    FiniteAbelianGroupElement,
    FiniteAbelianProductGroup,
)


def _run(
    moduli: tuple[int, ...], coordinates: tuple[tuple[int, ...], ...]
) -> FiniteAbelianSubsetSumResult:
    group = FiniteAbelianProductGroup(moduli=moduli)
    sequence = tuple(
        FiniteAbelianGroupElement(group=group, coordinates=coordinate)
        for coordinate in coordinates
    )
    return finite_abelian_subset_sum_profile(group, sequence)


def _counts(result: FiniteAbelianSubsetSumResult) -> dict[tuple[int, ...], int]:
    return {row.element.coordinates: row.multiplicity for row in result.rows}


def test_klein_four_fixture_covers_every_element_once() -> None:
    result = _run((2, 2), ((1, 0), (0, 1)))

    assert _counts(result) == dict.fromkeys(product(range(2), repeat=2), 1)
    assert result.support_size == 4
    assert result.covers_group
    assert result.total_subsets == 4


def test_empty_and_repeated_indexed_elements_preserve_multiplicity() -> None:
    empty = _run((3,), ())
    assert _counts(empty) == {(0,): 1, (1,): 0, (2,): 0}
    assert empty.support_size == 1
    assert not empty.covers_group

    repeated = _run((2,), ((1,), (1,)))
    assert _counts(repeated) == {(0,): 2, (1,): 2}
    assert repeated.total_subsets == 4


def test_small_profiles_match_indexed_subset_enumeration() -> None:
    for moduli, source in (
        ((2,), ((1,), (0,), (1,))),
        ((3,), ((1,), (2,))),
        ((2, 2), ((1, 1), (1, 0), (0, 1))),
    ):
        result = _run(moduli, source)
        expected = dict.fromkeys(product(*(range(m) for m in moduli)), 0)
        for mask in range(1 << len(source)):
            total = tuple(
                sum(
                    source[index][axis]
                    for index in range(len(source))
                    if mask & (1 << index)
                )
                % modulus
                for axis, modulus in enumerate(moduli)
            )
            expected[total] += 1
        assert _counts(result) == expected


def test_group_binding_and_result_round_trip() -> None:
    group = FiniteAbelianProductGroup(moduli=(2, 3))
    other = FiniteAbelianProductGroup(moduli=(6,))
    element = FiniteAbelianGroupElement(group=other, coordinates=(1,))
    with pytest.raises(ValueError, match="supplied group"):
        FiniteAbelianSubsetSumRequest(group=group, sequence=(element,))

    result = _run((2, 3), ((1, 1), (1, 2)))
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_order_bound_is_accepted_with_complete_zero_profile() -> None:
    result = _run((4096,), ())

    assert len(result.rows) == 4096
    assert result.rows[0].multiplicity == 1
    assert result.support_size == 1


def test_transition_bound_is_admitted_before_dynamic_programming() -> None:
    group = FiniteAbelianProductGroup(moduli=(1024,))
    zero = FiniteAbelianGroupElement(group=group, coordinates=(0,))
    with pytest.raises(OperationResourceAdmissionError, match="transition bound"):
        finite_abelian_subset_sum_profile(group, (zero,) * 4095)
