"""Tests for symbolic dynamics operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.symbolic_dynamics._models import (
    AdjacencyShiftRequest,
    BlockLanguageRequest,
    FiniteTypeShiftRequest,
    HigherBlockRequest,
    PeriodicPointProfileRequest,
)
from jacobian.math.symbolic_dynamics._operations import (
    compute_block_language,
    compute_higher_block,
    compute_periodic_point_profile,
    construct_adjacency_shift,
    construct_finite_type_shift,
)


class TestFiniteTypeShift:
    def test_golden_mean(self):
        req = FiniteTypeShiftRequest(alphabet=("0", "1"), forbidden_blocks=(("1", "1"),))
        result = construct_finite_type_shift(req)
        assert result.max_forbidden_length == 2
        assert not result.is_empty
        assert result.num_states > 0

    def test_empty_shift(self):
        req = FiniteTypeShiftRequest(alphabet=("0",), forbidden_blocks=((),))
        result = construct_finite_type_shift(req)
        assert result.is_empty

    def test_full_shift(self):
        req = FiniteTypeShiftRequest(alphabet=("0", "1"), forbidden_blocks=())
        result = construct_finite_type_shift(req)
        assert not result.is_empty


class TestBlockLanguage:
    def test_golden_mean_blocks_2(self):
        req = BlockLanguageRequest(
            alphabet=("0", "1"), forbidden_blocks=(("1", "1"),), block_length=2,
        )
        result = compute_block_language(req)
        assert result.count == 3
        blocks = {b for b in result.allowed_blocks}
        assert ("0", "0") in blocks
        assert ("0", "1") in blocks
        assert ("1", "0") in blocks
        assert ("1", "1") not in blocks

    def test_full_shift_blocks_2(self):
        req = BlockLanguageRequest(
            alphabet=("0", "1"), forbidden_blocks=(), block_length=2,
        )
        result = compute_block_language(req)
        assert result.count == 4


class TestAdjacencyShift:
    def test_golden_mean_matrix(self):
        req = AdjacencyShiftRequest(matrix=((1, 1), (1, 0)))
        result = construct_adjacency_shift(req)
        assert result.is_irreducible
        assert result.is_mixing

    def test_not_essential(self):
        req = AdjacencyShiftRequest(matrix=((0, 0), (1, 0)))
        result = construct_adjacency_shift(req)
        assert not result.is_essential

    def test_not_irreducible(self):
        req = AdjacencyShiftRequest(matrix=((1, 0), (0, 1)))
        result = construct_adjacency_shift(req)
        assert not result.is_irreducible

    def test_rejects_non_square(self):
        with pytest.raises(ValidationError, match="square"):
            AdjacencyShiftRequest(matrix=((1, 0), (1,)))

    def test_rejects_negative(self):
        with pytest.raises(ValidationError, match="non-negative"):
            AdjacencyShiftRequest(matrix=((-1, 0), (0, 1)))


class TestPeriodicPointProfile:
    def test_golden_mean(self):
        req = PeriodicPointProfileRequest(matrix=((1, 1), (1, 0)), max_period=5)
        result = compute_periodic_point_profile(req)
        assert result.fix_counts[0] == 1
        assert result.fix_counts[1] == 3

    def test_exact_counts_positive(self):
        req = PeriodicPointProfileRequest(matrix=((1, 1), (1, 0)), max_period=5)
        result = compute_periodic_point_profile(req)
        for count in result.exact_counts:
            assert count >= 0


class TestHigherBlock:
    def test_higher_block_2(self):
        req = HigherBlockRequest(
            alphabet=("0", "1"), forbidden_blocks=(("1", "1"),), n=2,
        )
        result = compute_higher_block(req)
        assert result.n == 2
        assert len(result.new_alphabet) == 2

    def test_rejects_n_too_small(self):
        with pytest.raises(ValidationError):
            HigherBlockRequest(alphabet=("0", "1"), forbidden_blocks=(), n=1)
