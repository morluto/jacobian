"""Tests for RSK permutation operation."""

import pytest
from pydantic import ValidationError

from jacobian.math.algebraic_combinatorics._models import RSKPermutationRequest
from jacobian.math.algebraic_combinatorics._operations import compute_rsk_permutation


class TestRSK:
    def test_empty(self) -> None:
        result = compute_rsk_permutation(RSKPermutationRequest(permutation=()))
        assert result.shape == ()
        assert result.lis_length == 0
        assert result.lds_length == 0

    def test_identity(self) -> None:
        result = compute_rsk_permutation(
            RSKPermutationRequest(permutation=(1, 2, 3, 4, 5))
        )
        assert result.shape == (5,)
        assert result.lis_length == 5
        assert result.lds_length == 1
        assert result.p_tableau == ((1, 2, 3, 4, 5),)
        assert result.q_tableau == ((1, 2, 3, 4, 5),)

    def test_reverse(self) -> None:
        result = compute_rsk_permutation(
            RSKPermutationRequest(permutation=(5, 4, 3, 2, 1))
        )
        assert result.shape == (1, 1, 1, 1, 1)
        assert result.lis_length == 1
        assert result.lds_length == 5

    def test_132(self) -> None:
        result = compute_rsk_permutation(RSKPermutationRequest(permutation=(1, 3, 2)))
        assert result.shape == (2, 1)
        assert result.lis_length == 2
        assert result.lds_length == 2

    def test_312(self) -> None:
        result = compute_rsk_permutation(RSKPermutationRequest(permutation=(3, 1, 2)))
        assert result.shape == (2, 1)
        assert result.lis_length == 2
        assert result.lds_length == 2

    def test_invalid_not_permutation(self) -> None:
        with pytest.raises(ValidationError, match="permutation"):
            RSKPermutationRequest(permutation=(1, 2, 2))

    def test_p_and_q_same_shape(self) -> None:
        for perm in [(1, 2, 3), (3, 2, 1), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2)]:
            result = compute_rsk_permutation(RSKPermutationRequest(permutation=perm))
            p_shape = tuple(len(row) for row in result.p_tableau)
            q_shape = tuple(len(row) for row in result.q_tableau)
            assert p_shape == q_shape == result.shape

    def test_single_element(self) -> None:
        result = compute_rsk_permutation(RSKPermutationRequest(permutation=(1,)))
        assert result.shape == (1,)
        assert result.p_tableau == ((1,),)
        assert result.q_tableau == ((1,),)
