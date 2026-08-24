"""Source-bound incidence profiles and finite trade comparison tests."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.math.incidence_structures import (
    ContainmentProfileResult,
    IncidenceMomentComparison,
    IncidenceStructure,
    IncidenceTradeResult,
    check_incidence_trade,
    containment_profile,
)
from jacobian.math.incidence_structures._models import (
    MAX_RESULT_BYTES,
    MAX_TRADE_ORDER,
    ContainmentProfileRequest,
    IncidenceMultiplicityDifference,
    IncidenceTradeRequest,
)
from jacobian.math.incidence_structures._operations import compute_incidence_trade

_TRADE_POINTS = ("1", "3", "4", "5", "6", "7", "8", "9", "10", "11", "13", "20")
_REMOVED_BLOCKS = (
    ("1",),
    ("13",),
    ("20",),
    ("5", "7"),
    ("5", "8"),
    ("6", "10"),
    ("6", "11"),
    ("1", "3", "5"),
    ("1", "4", "6"),
    ("3", "4", "9"),
    ("5", "6", "9"),
    ("7", "8", "13"),
    ("10", "11", "20"),
)
_INSERTED_BLOCKS = (
    ("1", "5"),
    ("1", "6"),
    ("5", "6"),
    ("7", "13"),
    ("8", "13"),
    ("10", "20"),
    ("11", "20"),
    ("1", "3", "4"),
    ("3", "5", "9"),
    ("4", "6", "9"),
    ("5", "7", "8"),
    ("6", "10", "11"),
)


def _family(
    blocks: tuple[tuple[str, ...], ...],
    prefix: str,
    *,
    points: tuple[str, ...] = _TRADE_POINTS,
) -> IncidenceStructure:
    return IncidenceStructure(
        points=points,
        block_ids=tuple(f"{prefix}{index}" for index in range(len(blocks))),
        blocks=blocks,
    )


def test_published_thirteen_for_twelve_trade_matches_through_order_two() -> None:
    removed = _family(_REMOVED_BLOCKS, "r")
    inserted = _family(_INSERTED_BLOCKS, "a")

    removed_points = containment_profile(removed, 1)
    inserted_points = containment_profile(inserted, 1)
    removed_pairs = containment_profile(removed, 2)
    inserted_pairs = containment_profile(inserted, 2)

    assert removed_points.subset_profile == inserted_points.subset_profile
    assert removed_points.total_multiplicity == inserted_points.total_multiplicity == 29
    assert removed_pairs.subset_profile == inserted_pairs.subset_profile
    assert removed_pairs.total_multiplicity == inserted_pairs.total_multiplicity == 22
    assert removed_pairs.histogram == inserted_pairs.histogram == ((0, 44), (1, 22))

    result = check_incidence_trade(removed, inserted, 2)
    assert result.zeroth_difference == 1
    assert result.positive_moments_equal
    assert tuple(comparison.order for comparison in result.comparisons) == (1, 2)
    assert tuple(
        (comparison.left_total, comparison.right_total)
        for comparison in result.comparisons
    ) == ((29, 29), (22, 22))
    assert all(not comparison.differences for comparison in result.comparisons)


def test_distinct_indices_preserve_repeated_blocks() -> None:
    repeated = _family((("a",), ("a",)), "b", points=("a", "b"))

    result = containment_profile(repeated, 1)

    assert result.subset_profile == ((("a",), 2), (("b",), 0))
    assert result.histogram == ((0, 1), (2, 1))
    assert result.total_multiplicity == 2


def test_nontrade_returns_every_nonzero_difference_in_point_order() -> None:
    left = _family((("a",),), "l", points=("a", "b"))
    right = _family((("b",),), "r", points=("a", "b"))

    result = check_incidence_trade(left, right, 1)

    assert result.zeroth_difference == 0
    assert not result.positive_moments_equal
    comparison = result.comparisons[0]
    assert not comparison.equal
    assert tuple(
        (
            difference.subset,
            difference.left_multiplicity,
            difference.right_multiplicity,
        )
        for difference in comparison.differences
    ) == ((("a",), 1, 0), (("b",), 0, 1))


def test_second_order_difference_is_distinct_from_equal_first_moments() -> None:
    left = _family((("a", "b"),), "l", points=("a", "b"))
    right = _family((("a",), ("b",)), "r", points=("a", "b"))

    result = check_incidence_trade(left, right, 2)

    assert result.zeroth_difference == -1
    assert not result.positive_moments_equal
    first, second = result.comparisons
    assert first.equal
    assert first.left_total == first.right_total == 2
    assert second.equal is False
    assert second.left_total == 1
    assert second.right_total == 0
    assert tuple(
        (
            difference.subset,
            difference.left_multiplicity,
            difference.right_multiplicity,
        )
        for difference in second.differences
    ) == ((("a", "b"), 1, 0),)


def test_third_order_trade_is_admitted_when_budgets_fit() -> None:
    left = _family(
        (("a", "b"), ("a", "c"), ("b", "c")),
        "l",
        points=("a", "b", "c"),
    )
    right = _family(
        (("a", "b", "c"), ("a",), ("b",), ("c",)),
        "r",
        points=("a", "b", "c"),
    )

    result = check_incidence_trade(left, right, 3)

    assert result.zeroth_difference == -1
    assert not result.positive_moments_equal
    first, second, third = result.comparisons
    assert tuple(comparison.order for comparison in result.comparisons) == (1, 2, 3)
    assert first.equal
    assert first.left_total == first.right_total == 6
    assert second.equal
    assert second.left_total == second.right_total == 3
    assert third.left_total == 0
    assert third.right_total == 1
    assert tuple(
        (
            difference.subset,
            difference.left_multiplicity,
            difference.right_multiplicity,
        )
        for difference in third.differences
    ) == ((("a", "b", "c"), 0, 1),)


def test_empty_fixed_order_profile_has_explicit_zero_convention() -> None:
    incidence = _family((("a", "b"),), "b", points=("a", "b"))

    result = containment_profile(incidence, 3)

    assert result.subset_profile == ()
    assert result.histogram == ()
    assert result.total_multiplicity == 0
    assert result.min_multiplicity == result.max_multiplicity == 0
    assert result.is_constant
    assert result.constant_lambda == 0


def test_profile_admission_uses_the_complete_subset_count() -> None:
    accepted_points = tuple(f"p{index}" for index in range(32))
    accepted = _family(((),), "b", points=accepted_points)
    assert ContainmentProfileRequest(incidence=accepted, t=3).t == 3

    rejected_points = tuple(f"p{index}" for index in range(33))
    rejected = _family(((),), "b", points=rejected_points)
    with pytest.raises(ValidationError, match="subset-count budget"):
        ContainmentProfileRequest(incidence=rejected, t=3)


def test_profile_admission_reserves_output_for_repeated_labels() -> None:
    points = tuple(f"p{index}-" + "x" * 1_000 for index in range(100))
    incidence = _family(((),), "b", points=points)

    with pytest.raises(ValidationError, match="output budget"):
        ContainmentProfileRequest(incidence=incidence, t=2)


def test_trade_admission_is_budget_derived_with_conservative_order_ceiling() -> None:
    points = tuple(f"p{index}" for index in range(33))
    left = _family(((),), "l", points=points)
    right = _family(((),), "r", points=points)

    assert IncidenceTradeRequest(left=left, right=right, max_order=2)

    with pytest.raises(ValidationError, match="subset-count budget"):
        IncidenceTradeRequest(left=left, right=right, max_order=3)

    tiny_left = _family((("a",),), "l", points=("a", "b"))
    tiny_right = _family((("b",),), "r", points=("a", "b"))
    assert IncidenceTradeRequest(
        left=tiny_left,
        right=tiny_right,
        max_order=MAX_TRADE_ORDER,
    )
    with pytest.raises(ValidationError, match="less than or equal"):
        IncidenceTradeRequest(
            left=tiny_left,
            right=tiny_right,
            max_order=MAX_TRADE_ORDER + 1,
        )


def test_trade_requires_identical_ordered_point_parents() -> None:
    left = _family((("a",),), "l", points=("a", "b"))
    right = _family((("a",),), "r", points=("b", "a"))

    with pytest.raises(ValidationError, match="same ordered point axis"):
        IncidenceTradeRequest(left=left, right=right, max_order=1)


def _long_id_family(prefix: str, filler: str, id_length: int) -> IncidenceStructure:
    return IncidenceStructure(
        points=("a",),
        block_ids=tuple(
            f"{prefix}{index}-" + filler * (id_length - len(f"{prefix}{index}-"))
            for index in range(100)
        ),
        blocks=((),) * 100,
    )


def test_trade_admission_reserves_output_for_every_source_echo() -> None:
    left = _long_id_family("l", "x", 3_000)
    right = _long_id_family("r", "y", 3_000)

    with pytest.raises(ValidationError, match="output budget"):
        IncidenceTradeRequest(left=left, right=right, max_order=1)


def test_admitted_trade_returns_typed_result_within_output_budget() -> None:
    left = _long_id_family("l", "x", 1_400)
    right = _long_id_family("r", "y", 1_400)

    result = check_incidence_trade(left, right, 1)

    assert result.positive_moments_equal
    assert all(comparison.equal for comparison in result.comparisons)
    assert len(result.model_dump_json()) <= MAX_RESULT_BYTES


def test_exported_native_values_compare_equal_across_member_orders() -> None:
    def family(blocks: tuple[tuple[str, ...], ...]) -> IncidenceStructure:
        return IncidenceStructure(
            points=("a", "b"),
            block_ids=("x", "y"),
            blocks=blocks,
        )

    left = family((("a", "b"), ("a",)))
    right = family((("b", "a"), ("a",)))

    profile = containment_profile(left, 2)
    trade = check_incidence_trade(left, right, 2)

    assert left.blocks == (("a", "b"), ("a",))
    assert left == right
    assert profile.incidence == left
    assert trade.left == trade.right
    serialized = trade.model_dump(mode="json")
    assert serialized["left"] == serialized["right"]
    assert trade.positive_moments_equal
    assert all(comparison.equal for comparison in trade.comparisons)


def test_result_validation_accepts_reordered_source_members() -> None:
    incidence = _family((("b", "a"),), "b", points=("a", "b"))
    result = containment_profile(incidence, 1)
    payload: dict[str, Any] = result.model_dump(mode="python")
    reordered = _family((("a", "b"),), "b", points=("a", "b"))
    payload["incidence"] = reordered

    accepted = ContainmentProfileResult.model_validate(payload)

    assert accepted == result


def test_profile_result_replays_source_and_authoritative_totals() -> None:
    incidence = _family((("a",), ("a", "b")), "b", points=("a", "b"))
    result = containment_profile(incidence, 1)
    payload: dict[str, Any] = result.model_dump(mode="python")
    payload["total_multiplicity"] = result.total_multiplicity + 1

    with pytest.raises(ValidationError, match="does not match"):
        ContainmentProfileResult.model_validate(payload)

    changed_source = _family((("b",), ("a", "b")), "b", points=("a", "b"))
    payload = result.model_dump(mode="python")
    payload["incidence"] = changed_source
    with pytest.raises(ValidationError, match="does not match"):
        ContainmentProfileResult.model_validate(payload)


def test_trade_result_replays_sources_and_zeroth_difference() -> None:
    left = _family((("a",), ("b",)), "l", points=("a", "b"))
    right = _family((("a", "b"),), "r", points=("a", "b"))
    result = check_incidence_trade(left, right, 1)
    payload: dict[str, Any] = result.model_dump(mode="python")
    payload["zeroth_difference"] = 0

    with pytest.raises(ValidationError, match="does not match"):
        IncidenceTradeResult.model_validate(payload)


def test_moment_totals_bind_to_sparse_differences() -> None:
    left = _family((("a", "b"), ("b",)), "l", points=("a", "b"))
    right = _family((("a", "b"), ("a",)), "r", points=("a", "b"))

    result = check_incidence_trade(left, right, 1)

    comparison = result.comparisons[0]
    assert not comparison.equal
    assert comparison.points == ("a", "b")
    assert (comparison.left_total, comparison.right_total) == (3, 3)
    assert tuple(
        (
            difference.subset,
            difference.left_multiplicity - difference.right_multiplicity,
        )
        for difference in comparison.differences
    ) == ((("a",), -1), (("b",), 1))


def test_equal_empty_moment_comparison_is_accepted_standalone() -> None:
    left = _family((("a",),), "l", points=("a", "b"))
    right = _family((("a",),), "r", points=("a", "b"))

    comparison = IncidenceMomentComparison(
        left=left,
        right=right,
        points=("a", "b"),
        order=1,
        left_total=1,
        right_total=1,
        differences=(),
        equal=True,
    )

    assert comparison.left_total == comparison.right_total == 1


def test_zero_residual_moment_comparison_is_accepted_standalone() -> None:
    left = _family((("a", "b"), ("a", "b")), "l", points=("a", "b"))
    right = _family((("a", "b"),), "r", points=("a", "b"))

    comparison = IncidenceMomentComparison(
        left=left,
        right=right,
        points=("a", "b"),
        order=2,
        left_total=2,
        right_total=1,
        differences=(
            IncidenceMultiplicityDifference(
                subset=("a", "b"),
                left_multiplicity=2,
                right_multiplicity=1,
            ),
        ),
        equal=False,
    )

    assert (comparison.left_total, comparison.right_total) == (2, 1)


def test_saturated_sparse_multiplicities_are_accepted_with_witness_families() -> None:
    left = _family((("a",),) * 100, "l", points=("a", "b"))
    right = _family(((),), "r", points=("a", "b"))

    comparison = IncidenceMomentComparison(
        left=left,
        right=right,
        points=("a", "b"),
        order=1,
        left_total=100,
        right_total=0,
        differences=(
            IncidenceMultiplicityDifference(
                subset=("a",),
                left_multiplicity=100,
                right_multiplicity=0,
            ),
        ),
        equal=False,
    )

    assert (comparison.left_total, comparison.right_total) == (100, 0)


def test_forged_totals_diverging_from_retained_profiles_are_rejected() -> None:
    left = _family((("a", "b"),), "l", points=("a", "b"))
    right = _family(((),), "r", points=("a", "b"))

    with pytest.raises(
        ValidationError,
        match="does not match the retained incidence families",
    ):
        IncidenceMomentComparison(
            left=left,
            right=right,
            points=("a", "b"),
            order=2,
            left_total=2,
            right_total=1,
            differences=(
                IncidenceMultiplicityDifference(
                    subset=("a", "b"),
                    left_multiplicity=2,
                    right_multiplicity=0,
                ),
            ),
            equal=False,
        )


def test_forged_positive_totals_without_order_subsets_are_rejected() -> None:
    left = _family((("a",),), "l", points=("a",))
    right = _family((("a",),), "r", points=("a",))

    with pytest.raises(ValidationError, match="moment totals do not match"):
        IncidenceMomentComparison(
            left=left,
            right=right,
            points=("a",),
            order=2,
            left_total=1,
            right_total=1,
            differences=(),
            equal=True,
        )


def test_forged_saturated_shared_core_differences_are_rejected() -> None:
    left = _family((("a", "b"),) * 100, "l", points=("a", "b", "c"))
    right = _family(((),), "r", points=("a", "b", "c"))

    with pytest.raises(
        ValidationError,
        match="does not match the retained incidence families",
    ):
        IncidenceMomentComparison(
            left=left,
            right=right,
            points=("a", "b", "c"),
            order=2,
            left_total=200,
            right_total=0,
            differences=(
                IncidenceMultiplicityDifference(
                    subset=("a", "b"),
                    left_multiplicity=100,
                    right_multiplicity=0,
                ),
                IncidenceMultiplicityDifference(
                    subset=("a", "c"),
                    left_multiplicity=100,
                    right_multiplicity=0,
                ),
            ),
            equal=False,
        )


def test_forged_right_saturated_shared_core_differences_are_rejected() -> None:
    right = _family((("a", "b"),) * 100, "r", points=("a", "b", "c"))
    left = _family(((),), "l", points=("a", "b", "c"))

    with pytest.raises(
        ValidationError,
        match="does not match the retained incidence families",
    ):
        IncidenceMomentComparison(
            left=left,
            right=right,
            points=("a", "b", "c"),
            order=2,
            left_total=0,
            right_total=200,
            differences=(
                IncidenceMultiplicityDifference(
                    subset=("a", "b"),
                    left_multiplicity=0,
                    right_multiplicity=100,
                ),
                IncidenceMultiplicityDifference(
                    subset=("a", "c"),
                    left_multiplicity=0,
                    right_multiplicity=100,
                ),
            ),
            equal=False,
        )


def test_shared_core_boundary_is_witnessed_by_retained_families() -> None:
    left = _family(
        (("a", "b", "c"),) * 60 + (("a", "b"),) * 40, "l", points=("a", "b", "c")
    )
    right = _family((("b", "c"),) * 60, "r", points=("a", "b", "c"))

    comparison = IncidenceMomentComparison(
        left=left,
        right=right,
        points=("a", "b", "c"),
        order=2,
        left_total=220,
        right_total=60,
        differences=(
            IncidenceMultiplicityDifference(
                subset=("a", "b"),
                left_multiplicity=100,
                right_multiplicity=0,
            ),
            IncidenceMultiplicityDifference(
                subset=("a", "c"),
                left_multiplicity=60,
                right_multiplicity=0,
            ),
        ),
        equal=False,
    )

    assert (comparison.left_total, comparison.right_total) == (220, 60)


def test_forged_unwitnessable_shared_core_totals_are_rejected() -> None:
    left = _family(
        (("a", "b", "c"),) * 60 + (("a", "b"),) * 40, "l", points=("a", "b", "c")
    )
    right = _family((("b", "c"),) * 60, "r", points=("a", "b", "c"))

    with pytest.raises(ValidationError, match="moment totals do not match"):
        IncidenceMomentComparison(
            left=left,
            right=right,
            points=("a", "b", "c"),
            order=2,
            left_total=219,
            right_total=59,
            differences=(
                IncidenceMultiplicityDifference(
                    subset=("a", "b"),
                    left_multiplicity=100,
                    right_multiplicity=0,
                ),
                IncidenceMultiplicityDifference(
                    subset=("a", "c"),
                    left_multiplicity=60,
                    right_multiplicity=0,
                ),
            ),
            equal=False,
        )


def test_disjoint_saturated_keys_are_accepted_with_witness_families() -> None:
    left = _family(
        (("a", "b"),) * 50 + (("c", "d"),) * 50, "l", points=("a", "b", "c", "d")
    )
    right = _family(((),), "r", points=("a", "b", "c", "d"))

    comparison = IncidenceMomentComparison(
        left=left,
        right=right,
        points=("a", "b", "c", "d"),
        order=2,
        left_total=100,
        right_total=0,
        differences=(
            IncidenceMultiplicityDifference(
                subset=("a", "b"),
                left_multiplicity=50,
                right_multiplicity=0,
            ),
            IncidenceMultiplicityDifference(
                subset=("c", "d"),
                left_multiplicity=50,
                right_multiplicity=0,
            ),
        ),
        equal=False,
    )

    assert (comparison.left_total, comparison.right_total) == (100, 0)


def test_forged_unrealizable_sparse_zero_profile_is_rejected() -> None:
    paired_points = ("a", "b", "c", "d", "e", "f", "g", "h")
    left = _family((paired_points,) * 60, "l", points=paired_points)
    right = _family((("a", "c", "e", "g"),) * 100, "r", points=paired_points)

    with pytest.raises(
        ValidationError,
        match="does not match the retained incidence families",
    ):
        IncidenceMomentComparison(
            left=left,
            right=right,
            points=paired_points,
            order=2,
            left_total=960,
            right_total=720,
            differences=tuple(
                IncidenceMultiplicityDifference(
                    subset=subset,
                    left_multiplicity=60,
                    right_multiplicity=0,
                )
                for subset in (("a", "b"), ("c", "d"), ("e", "f"), ("g", "h"))
            ),
            equal=False,
        )

    witnessed = IncidenceMomentComparison(
        left=_family((("a", "b", "c", "d"),) * 25, "l", points=paired_points),
        right=_family(
            (("a", "c"),) * 25
            + (("a", "d"),) * 25
            + (("b", "c"),) * 25
            + (("b", "d"),) * 25,
            "r",
            points=paired_points,
        ),
        points=paired_points,
        order=2,
        left_total=150,
        right_total=100,
        differences=(
            IncidenceMultiplicityDifference(
                subset=("a", "b"),
                left_multiplicity=25,
                right_multiplicity=0,
            ),
            IncidenceMultiplicityDifference(
                subset=("c", "d"),
                left_multiplicity=25,
                right_multiplicity=0,
            ),
        ),
        equal=False,
    )
    payload: dict[str, Any] = witnessed.model_dump(mode="python")
    payload["right_total"] = 720

    with pytest.raises(ValidationError, match="moment totals do not match"):
        IncidenceMomentComparison.model_validate(payload)


def test_forged_out_of_combination_order_difference_rows_are_rejected() -> None:
    left = _family((("a",), ("b",)), "l", points=("a", "b"))
    right = _family(((),), "r", points=("a", "b"))

    with pytest.raises(ValidationError, match="combination order"):
        IncidenceMomentComparison(
            left=left,
            right=right,
            points=("a", "b"),
            order=1,
            left_total=2,
            right_total=0,
            differences=(
                IncidenceMultiplicityDifference(
                    subset=("b",),
                    left_multiplicity=1,
                    right_multiplicity=0,
                ),
                IncidenceMultiplicityDifference(
                    subset=("a",),
                    left_multiplicity=1,
                    right_multiplicity=0,
                ),
            ),
            equal=False,
        )


def test_moment_comparison_round_trips_through_serialization() -> None:
    left = _family((("a", "b"), ("a", "b")), "l", points=("a", "b"))
    right = _family((("a", "b"),), "r", points=("a", "b"))

    comparison = IncidenceMomentComparison(
        left=left,
        right=right,
        points=("a", "b"),
        order=2,
        left_total=2,
        right_total=1,
        differences=(
            IncidenceMultiplicityDifference(
                subset=("a", "b"),
                left_multiplicity=2,
                right_multiplicity=1,
            ),
        ),
        equal=False,
    )

    payload: dict[str, Any] = comparison.model_dump(mode="python")
    assert IncidenceMomentComparison.model_validate(payload) == comparison
    serialized = comparison.model_dump(mode="json")
    assert serialized["left"] == left.model_dump(mode="json")
    assert serialized["right"] == right.model_dump(mode="json")


def test_forged_mutated_serialized_multiplicities_are_rejected() -> None:
    left = _family(
        (("a", "b"),) * 50 + (("c", "d"),) * 50, "l", points=("a", "b", "c", "d")
    )
    right = _family(((),), "r", points=("a", "b", "c", "d"))
    comparison = IncidenceMomentComparison(
        left=left,
        right=right,
        points=("a", "b", "c", "d"),
        order=2,
        left_total=100,
        right_total=0,
        differences=(
            IncidenceMultiplicityDifference(
                subset=("a", "b"),
                left_multiplicity=50,
                right_multiplicity=0,
            ),
            IncidenceMultiplicityDifference(
                subset=("c", "d"),
                left_multiplicity=50,
                right_multiplicity=0,
            ),
        ),
        equal=False,
    )

    payload: dict[str, Any] = comparison.model_dump(mode="json")
    payload["differences"][0]["left_multiplicity"] = 51

    with pytest.raises(
        ValidationError,
        match="does not match the retained incidence families",
    ):
        IncidenceMomentComparison.model_validate(payload)


def test_forged_repeated_labels_in_difference_values_are_rejected() -> None:
    with pytest.raises(ValidationError, match="distinct labels"):
        IncidenceMultiplicityDifference(
            subset=("a", "a"),
            left_multiplicity=1,
            right_multiplicity=0,
        )


def test_forged_equal_totals_with_sparse_differences_are_rejected() -> None:
    left = _family((("a", "b"),), "l", points=("a", "b"))
    right = _family((("a", "b"),), "r", points=("a", "b"))

    with pytest.raises(
        ValidationError,
        match="does not match the retained incidence families",
    ):
        IncidenceMomentComparison(
            left=left,
            right=right,
            points=("a", "b"),
            order=2,
            left_total=1,
            right_total=1,
            differences=(
                IncidenceMultiplicityDifference(
                    subset=("a", "b"),
                    left_multiplicity=1,
                    right_multiplicity=0,
                ),
            ),
            equal=False,
        )


def test_forged_mismatched_family_point_axes_are_rejected() -> None:
    left = _family((("a",),), "l", points=("a", "b"))
    right = _family((("a",),), "r", points=("b", "a"))

    with pytest.raises(ValidationError, match="share the declared ordered point axis"):
        IncidenceMomentComparison(
            left=left,
            right=right,
            points=("a", "b"),
            order=1,
            left_total=1,
            right_total=1,
            differences=(),
            equal=True,
        )


def test_forged_totals_below_sparse_differences_are_rejected() -> None:
    left = _family((("a", "b"), ("a", "b")), "l", points=("a", "b"))
    right = _family((("a", "b"),), "r", points=("a", "b"))

    with pytest.raises(ValidationError, match="moment totals do not match"):
        IncidenceMomentComparison(
            left=left,
            right=right,
            points=("a", "b"),
            order=2,
            left_total=1,
            right_total=0,
            differences=(
                IncidenceMultiplicityDifference(
                    subset=("a", "b"),
                    left_multiplicity=2,
                    right_multiplicity=1,
                ),
            ),
            equal=False,
        )


def test_forged_wrong_arity_subset_keys_are_rejected() -> None:
    left = _family((("a", "b"),), "l", points=("a", "b"))
    right = _family(((),), "r", points=("a", "b"))

    with pytest.raises(ValidationError, match="exactly order labels"):
        IncidenceMomentComparison(
            left=left,
            right=right,
            points=("a", "b"),
            order=2,
            left_total=1,
            right_total=0,
            differences=(
                IncidenceMultiplicityDifference(
                    subset=("a",),
                    left_multiplicity=1,
                    right_multiplicity=0,
                ),
            ),
            equal=False,
        )


def test_forged_repeated_label_subset_keys_are_rejected() -> None:
    left = _family((("a", "b"),), "l", points=("a", "b"))
    right = _family(((),), "r", points=("a", "b"))

    with pytest.raises(ValidationError, match="distinct labels"):
        IncidenceMomentComparison(
            left=left,
            right=right,
            points=("a", "b"),
            order=2,
            left_total=1,
            right_total=0,
            differences=(
                IncidenceMultiplicityDifference(
                    subset=("a", "a"),
                    left_multiplicity=1,
                    right_multiplicity=0,
                ),
            ),
            equal=False,
        )


def test_forged_undeclared_difference_labels_are_rejected() -> None:
    left = _family((("a", "b"),), "l", points=("a", "b"))
    right = _family(((),), "r", points=("a", "b"))

    with pytest.raises(ValidationError, match="declared point-axis labels"):
        IncidenceMomentComparison(
            left=left,
            right=right,
            points=("a", "b"),
            order=2,
            left_total=1,
            right_total=0,
            differences=(
                IncidenceMultiplicityDifference(
                    subset=("a", "z"),
                    left_multiplicity=1,
                    right_multiplicity=0,
                ),
            ),
            equal=False,
        )


def test_forged_out_of_axis_order_subset_keys_are_rejected() -> None:
    left = _family((("a", "b"),), "l", points=("a", "b"))
    right = _family(((),), "r", points=("a", "b"))

    with pytest.raises(ValidationError, match="point-axis order"):
        IncidenceMomentComparison(
            left=left,
            right=right,
            points=("a", "b"),
            order=2,
            left_total=1,
            right_total=0,
            differences=(
                IncidenceMultiplicityDifference(
                    subset=("b", "a"),
                    left_multiplicity=1,
                    right_multiplicity=0,
                ),
            ),
            equal=False,
        )


def test_forged_permuted_duplicate_subset_keys_are_rejected() -> None:
    left = _family((("a", "c"),), "l", points=("a", "b", "c"))
    right = _family(((),), "r", points=("a", "b", "c"))

    with pytest.raises(ValidationError, match="point-axis order"):
        IncidenceMomentComparison(
            left=left,
            right=right,
            points=("a", "b", "c"),
            order=2,
            left_total=1,
            right_total=0,
            differences=(
                IncidenceMultiplicityDifference(
                    subset=("a", "c"),
                    left_multiplicity=1,
                    right_multiplicity=0,
                ),
                IncidenceMultiplicityDifference(
                    subset=("c", "a"),
                    left_multiplicity=1,
                    right_multiplicity=0,
                ),
            ),
            equal=False,
        )


def test_forged_duplicate_subset_keys_are_rejected() -> None:
    left = _family((("a",),), "l", points=("a", "b"))
    right = _family(((),), "r", points=("a", "b"))

    with pytest.raises(ValidationError, match="unique"):
        IncidenceMomentComparison(
            left=left,
            right=right,
            points=("a", "b"),
            order=1,
            left_total=1,
            right_total=0,
            differences=(
                IncidenceMultiplicityDifference(
                    subset=("a",),
                    left_multiplicity=1,
                    right_multiplicity=0,
                ),
                IncidenceMultiplicityDifference(
                    subset=("a",),
                    left_multiplicity=1,
                    right_multiplicity=0,
                ),
            ),
            equal=False,
        )


def test_request_schema_exposes_validator_owned_trade_rules() -> None:
    schema = IncidenceTradeRequest.model_json_schema()
    assert "same ordered point axis" in schema["description"]
    assert (
        "Largest positive subset order"
        in schema["properties"]["max_order"]["description"]
    )


def test_trade_operation_is_discoverable_and_example_executes() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    operation = next(
        tool for tool in BUILTIN_TOOLS if tool.operation_id == "incidence.trade.check"
    )
    example = operation.examples[0]
    result = operation.run(operation.request_type.model_validate(example.input))
    assert result.positive_moments_equal
    assert result.zeroth_difference == 1


def test_catalog_adapter_returns_the_native_result() -> None:
    left = _family((("a",), ("b",)), "l", points=("a", "b"))
    right = _family((("a", "b"),), "r", points=("a", "b"))
    request = IncidenceTradeRequest(left=left, right=right, max_order=1)

    assert compute_incidence_trade(request) == check_incidence_trade(left, right, 1)
