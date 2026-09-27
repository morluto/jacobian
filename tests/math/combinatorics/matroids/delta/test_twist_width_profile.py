"""Complete bounded twist-width profiles for finite delta-matroids."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.matroids.delta import FiniteDeltaMatroid
from jacobian.math.combinatorics.matroids.delta.extra import (
    DeltaMatroidTwistWidthProfileRequest,
)
from jacobian.math.combinatorics.matroids.delta.extra_ops import twist_width_profile


def test_profile_matches_direct_symmetric_difference_for_every_twist() -> None:
    value = FiniteDeltaMatroid(
        ground=("a", "b", "c"),
        feasible=((0,), (0, 1), (0, 2), (1,), (1, 2), (2,)),
    )

    result = twist_width_profile(value)

    assert result.profile.ground == value.ground
    expected = []
    for mask in range(1 << len(value.ground)):
        twist = {index for index in range(len(value.ground)) if mask >> index & 1}
        sizes = [len(set(row) ^ twist) for row in value.feasible]
        expected.append(max(sizes) - min(sizes))
    assert result.profile.widths_by_mask == tuple(expected)
    assert result.profile.widths_by_mask == (1, 3, 3, 3, 3, 3, 3, 1)


def test_profile_preserves_mask_order_and_nonconstant_widths() -> None:
    value = FiniteDeltaMatroid(
        ground=("a", "b"),
        feasible=((), (0,), (0, 1)),
    )

    result = twist_width_profile(value)

    expected = []
    for mask in range(1 << len(value.ground)):
        twist = {index for index in range(len(value.ground)) if mask >> index & 1}
        sizes = [len(set(row) ^ twist) for row in value.feasible]
        expected.append(max(sizes) - min(sizes))
    assert result.profile.widths_by_mask == tuple(expected)
    assert result.profile.widths_by_mask == (2, 1, 1, 2)


def test_empty_ground_has_one_untwisted_state() -> None:
    value = FiniteDeltaMatroid(ground=(), feasible=((),))

    result = twist_width_profile(value)

    assert result.profile.widths_by_mask == (0,)


def test_profile_state_bound_rejects_before_exchange_work() -> None:
    value = FiniteDeltaMatroid(
        ground=tuple(f"e{i}" for i in range(13)),
        feasible=((),),
    )

    with pytest.raises(OperationResourceAdmissionError, match="state or work envelope"):
        twist_width_profile(value)


def test_profile_accepts_exact_state_limit() -> None:
    value = FiniteDeltaMatroid(
        ground=tuple(f"e{i}" for i in range(12)),
        feasible=((),),
    )

    result = twist_width_profile(value)

    assert len(result.profile.widths_by_mask) == 4_096
    assert set(result.profile.widths_by_mask) == {0}


def test_request_schema_publishes_complete_profile_bounds() -> None:
    schema = DeltaMatroidTwistWidthProfileRequest.model_json_schema()

    assert schema["admission_limits"] == {
        "max_subset_masks": 4_096,
        "max_mask_feasible_set_evaluations": 262_144,
    }
