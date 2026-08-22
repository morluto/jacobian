"""Tests for ordered-difference profile operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.additive_combinatorics._models import (
    OrderedDifferenceProfileRequest,
    OrderedDifferenceProfileResult,
)
from jacobian.math.additive_combinatorics._operations import (
    compute_ordered_difference_profile,
)


class TestOrderedDifferenceProfile:
    def test_three_points_2d(self):
        """Three points in Z^2 with known differences."""
        req = OrderedDifferenceProfileRequest(vectors=([0, 0], [1, 0], [0, 1]))
        result = compute_ordered_difference_profile(req)
        assert result.dimension == 2
        assert result.set_size == 3
        assert result.total_ordered_pairs == 6  # 3*2
        assert result.support_size > 0
        for entry in result.entries:
            assert entry.multiplicity == len(entry.pairs)

    def test_no_repeated(self):
        """A Sidon set has no repeated differences."""
        req = OrderedDifferenceProfileRequest(vectors=([0, 0, 0], [1, 0, 0], [0, 1, 0]))
        result = compute_ordered_difference_profile(req)
        for entry in result.entries:
            for pair in entry.pairs:
                assert pair.left_index != pair.right_index

    def test_repeated_difference(self):
        """Four points forming a parallelogram have repeated differences."""
        req = OrderedDifferenceProfileRequest(vectors=([0, 0], [1, 0], [0, 1], [1, 1]))
        result = compute_ordered_difference_profile(req)
        assert result.has_repeated_difference
        assert result.first_collision is not None
        assert result.max_multiplicity >= 2

    def test_total_pairs_formula(self):
        """Total ordered pairs must equal |A|(|A|-1)."""
        vectors = ([0, 0], [1, 0], [2, 0], [3, 0])
        req = OrderedDifferenceProfileRequest(vectors=tuple(v for v in vectors))
        result = compute_ordered_difference_profile(req)
        assert result.total_ordered_pairs == 4 * 3

    def test_single_point(self):
        """A single point has no differences."""
        req = OrderedDifferenceProfileRequest(vectors=([1, 2],))
        result = compute_ordered_difference_profile(req)
        assert result.total_ordered_pairs == 0
        assert result.support_size == 0
        assert result.entries == ()
        assert not result.has_repeated_difference

    def test_one_dimensional(self):
        """One-dimensional vectors work correctly."""
        req = OrderedDifferenceProfileRequest(vectors=([0], [1], [2]))
        result = compute_ordered_difference_profile(req)
        assert result.dimension == 1
        assert result.total_ordered_pairs == 6  # 3*2

    def test_mismatched_dimensions_rejected(self):
        """Vectors with different dimensions should raise."""
        with pytest.raises(ValueError):
            OrderedDifferenceProfileRequest(vectors=([0, 0], [1]))

    def test_translation_invariance(self):
        base = compute_ordered_difference_profile(
            OrderedDifferenceProfileRequest(vectors=([0, 0], [1, 0], [0, 1])),
        )
        shifted = compute_ordered_difference_profile(
            OrderedDifferenceProfileRequest(vectors=([5, -3], [6, -3], [5, -2])),
        )
        base_diffs = {e.difference: e.multiplicity for e in base.entries}
        shifted_diffs = {e.difference: e.multiplicity for e in shifted.entries}
        assert base_diffs == shifted_diffs

    def test_sign_reversal_symmetry(self):
        result = compute_ordered_difference_profile(
            OrderedDifferenceProfileRequest(vectors=([0, 0], [1, 0], [0, 1], [1, 1])),
        )
        mult = {e.difference: e.multiplicity for e in result.entries}
        assert mult
        for d, count in mult.items():
            assert mult[tuple(-c for c in d)] == count

    def test_result_replays_every_difference_from_source(self):
        req = OrderedDifferenceProfileRequest(vectors=([0, 0], [1, 0], [1, 1], [0, 1]))
        result = compute_ordered_difference_profile(req)
        seen = set()
        for entry in result.entries:
            for pair in entry.pairs:
                replayed = tuple(
                    result.vectors[pair.left_index][k]
                    - result.vectors[pair.right_index][k]
                    for k in range(result.dimension)
                )
                assert replayed == entry.difference
                seen.add((pair.left_index, pair.right_index))
        n = result.set_size
        assert seen == {(i, j) for i in range(n) for j in range(n) if i != j}

    def test_result_rejects_forged_difference(self):
        req = OrderedDifferenceProfileRequest(vectors=([0], [1]))
        result = compute_ordered_difference_profile(req)
        payload = result.model_dump()
        payload["entries"][-1]["difference"] = [2]
        with pytest.raises(ValidationError, match="pair difference must match vectors"):
            OrderedDifferenceProfileResult.model_validate(payload)

    def test_result_rejects_mutated_source(self):
        req = OrderedDifferenceProfileRequest(vectors=([0, 0], [1, 0], [1, 1], [0, 1]))
        result = compute_ordered_difference_profile(req)
        payload = result.model_dump()
        payload["vectors"] = [[9, 9], *list(payload["vectors"][1:])]
        with pytest.raises(ValidationError, match="pair difference must match vectors"):
            OrderedDifferenceProfileResult.model_validate(payload)
