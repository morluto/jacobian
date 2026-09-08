"""Exact finite abelian indexed subset-sum profiles."""

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.additive._subset_sum_residue import (
    SubsetSumResidueProfileRequest,
    SubsetSumResidueProfileResult,
    subset_sum_residue_profile,
)
from jacobian.math.combinatorics.additive.finite_abelian_subset_sum._models import (
    MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS,
    MAX_FINITE_ABELIAN_SUBSET_SUM_ORDER,
    bounded_finite_abelian_subset_sum_order,
)
from jacobian.math.combinatorics.additive.finite_abelian_subset_sum.operations import (
    finite_abelian_subset_sum_profile,
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


def test_bounded_order_short_circuits_rank_and_running_product() -> None:
    high_rank = FiniteAbelianProductGroup.model_construct(
        moduli=(2,) * (MAX_FINITE_ABELIAN_SUBSET_SUM_ORDER + 1)
    )
    assert bounded_finite_abelian_subset_sum_order(high_rank) is None
    binary = FiniteAbelianProductGroup(moduli=(2,) * MAX_FINITE_ABELIAN_SUBSET_SUM_ORDER)
    assert bounded_finite_abelian_subset_sum_order(binary) is None
    assert (
        bounded_finite_abelian_subset_sum_order(FiniteAbelianProductGroup(moduli=(2, 2)))
        == 4
    )


def test_overlong_sequence_rejects_before_group_binding_scan() -> None:
    group = FiniteAbelianProductGroup(moduli=(2,))
    other = FiniteAbelianProductGroup(moduli=(4,))
    mismatched = FiniteAbelianGroupElement(group=other, coordinates=(1,))
    with pytest.raises(OperationResourceAdmissionError) as caught:
        finite_abelian_subset_sum_profile(
            group, (mismatched,) * (MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS + 1)
        )
    assert caught.value.errors()[0]["type"] == (
        "additive.finite_abelian_subset_sum.input_length"
    )


def test_forged_result_rejects_high_rank_before_group_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(self: FiniteAbelianProductGroup) -> int:
        raise AssertionError("group.order must not be evaluated")

    monkeypatch.setattr(FiniteAbelianProductGroup, "order", property(explode))
    moduli = (2,) * (MAX_FINITE_ABELIAN_SUBSET_SUM_ORDER + 1)
    group = FiniteAbelianProductGroup(moduli=moduli)
    zero = FiniteAbelianGroupElement(group=group, coordinates=(0,) * len(moduli))
    with pytest.raises(ValidationError, match="group order exceeds"):
        SubsetSumResidueProfileResult.model_validate(
            {
                "source": {"items": []},
                "modulus": 1,
                "include_empty_subset": True,
                "include_witnesses": False,
                "residue_counts": [],
                "group": group,
                "sequence": (zero,),
                "group_rows": ({"element": zero, "multiplicity": 1},),
                "support_size": 1,
                "covers_group": False,
            }
        )


def test_forged_result_rejects_oversized_order_before_full_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(self: FiniteAbelianProductGroup) -> int:
        raise AssertionError("group.order must not be evaluated")

    monkeypatch.setattr(FiniteAbelianProductGroup, "order", property(explode))
    group = FiniteAbelianProductGroup(moduli=(2,) * 13)
    zero = FiniteAbelianGroupElement(group=group, coordinates=(0,) * 13)
    with pytest.raises(ValidationError, match="group order exceeds"):
        SubsetSumResidueProfileResult.model_validate(
            {
                "source": {"items": []},
                "modulus": 1,
                "include_empty_subset": True,
                "include_witnesses": False,
                "residue_counts": [],
                "group": group,
                "sequence": (),
                "group_rows": ({"element": zero, "multiplicity": 1},),
                "support_size": 1,
                "covers_group": False,
            }
        )
