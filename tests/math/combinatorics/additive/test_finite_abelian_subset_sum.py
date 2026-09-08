"""Exact finite abelian indexed subset-sum profiles."""

from itertools import product

import pytest

from jacobian.math.combinatorics.additive._subset_sum_residue import (
    SubsetSumResidueProfileRequest,
    SubsetSumResidueProfileResult,
    subset_sum_residue_profile,
)
from jacobian.math.groups.finite_abelian import (
    FiniteAbelianGroupElement,
    FiniteAbelianProductGroup,
)


def _run(
    moduli: tuple[int, ...], coordinates: tuple[tuple[int, ...], ...]
) -> SubsetSumResidueProfileResult:
    group = FiniteAbelianProductGroup(moduli=moduli)
    sequence = tuple(
        FiniteAbelianGroupElement(group=group, coordinates=coordinate)
        for coordinate in coordinates
    )
    return subset_sum_residue_profile(None, None, True, group=group, sequence=sequence)


def _counts(result: SubsetSumResidueProfileResult) -> dict[tuple[int, ...], int]:
    assert result.group_rows is not None
    return {row.element.coordinates: row.multiplicity for row in result.group_rows}


def test_klein_four_fixture_covers_every_element_once() -> None:
    result = _run((2, 2), ((1, 0), (0, 1)))

    assert _counts(result) == dict.fromkeys(product(range(2), repeat=2), 1)
    assert result.support_size == 4
    assert result.covers_group is True
    assert _counts(_run((4,), ((1,), (2,)))) == {(i,): 1 for i in range(4)}
    assert result.group_rows is not None
    assert sum(row.multiplicity > 0 for row in result.group_rows) == 4


def test_empty_and_repeated_indexed_elements_preserve_multiplicity() -> None:
    empty = _run((3,), ())
    assert _counts(empty) == {(0,): 1, (1,): 0, (2,): 0}

    repeated = _run((2,), ((1,), (1,)))
    assert _counts(repeated) == {(0,): 2, (1,): 2}
    assert sum(_counts(repeated).values()) == 4


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
        SubsetSumResidueProfileRequest(
            group=group, sequence=(element,), include_empty_subset=True
        )

    result = _run((2, 3), ((1, 1), (1, 2)))
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_order_bound_is_accepted_with_complete_zero_profile() -> None:
    result = _run((4096,), ())

    assert result.group_rows is not None
    assert len(result.group_rows) == 4096
    assert result.group_rows[0].multiplicity == 1
    assert sum(row.multiplicity for row in result.group_rows) == 1


def test_trivial_group_and_empty_subset_option() -> None:
    group = FiniteAbelianProductGroup(moduli=())
    zero = FiniteAbelianGroupElement(group=group, coordinates=())
    result = subset_sum_residue_profile(
        None, None, True, group=group, sequence=(zero, zero)
    )
    assert result.group_rows is not None
    assert result.group_rows[0].element.coordinates == ()
    assert result.group_rows[0].multiplicity == 4
    nonempty = subset_sum_residue_profile(
        None, None, False, group=group, sequence=(zero, zero)
    )
    assert nonempty.group_rows is not None
    assert nonempty.group_rows[0].multiplicity == 3


def test_zero_source_compression_accepts_large_indexed_input() -> None:
    group = FiniteAbelianProductGroup(moduli=(1024,))
    zero = FiniteAbelianGroupElement(group=group, coordinates=(0,))
    result = subset_sum_residue_profile(
        None, None, True, group=group, sequence=(zero,) * 4095
    )
    assert result.group_rows is not None
    assert result.group_rows[0].multiplicity == 1 << 4095


@pytest.mark.parametrize("source", [(), ((1,),), ((0,), (1,), (1,))])
def test_nonempty_profile_matches_indexed_enumeration(
    source: tuple[tuple[int, ...], ...],
) -> None:
    group = FiniteAbelianProductGroup(moduli=(3,))
    sequence = tuple(
        FiniteAbelianGroupElement(group=group, coordinates=c) for c in source
    )
    result = subset_sum_residue_profile(
        None, None, False, group=group, sequence=sequence
    )
    expected = dict.fromkeys(((0,), (1,), (2,)), 0)
    for mask in range(1, 1 << len(source)):
        expected[
            (sum(source[i][0] for i in range(len(source)) if mask >> i & 1) % 3,)
        ] += 1
    assert _counts(result) == expected
    assert (
        SubsetSumResidueProfileResult.model_validate_json(result.model_dump_json())
        == result
    )
