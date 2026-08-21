"""Tests for ordered-difference profile operations."""

import pytest

from jacobian.math.additive_combinatorics._models import (
    OrderedDifferenceProfileRequest,
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
