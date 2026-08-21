"""Tests for nonlinear binary code operations."""

from jacobian.math.code_nonlinear._models import (
    ConstantWeightProfileRequest,
    ConstantWeightRequest,
    ExplicitProfileRequest,
    ToSetSystemRequest,
    WordDistanceRequest,
)
from jacobian.math.code_nonlinear._operations import (
    compute_constant_weight,
    compute_constant_weight_profile,
    compute_distance_profile,
    compute_explicit_profile,
    compute_to_set_system,
    compute_word_distance,
)


class TestWordDistance:
    def test_identical_words(self) -> None:
        result = compute_word_distance(
            WordDistanceRequest(word1=(1, 0, 1), word2=(1, 0, 1))
        )
        assert result.distance == 0
        assert result.differing_coordinates == ()
        assert result.weight1 == 2
        assert result.weight2 == 2
        assert result.support_intersection == 2

    def test_complementary_words(self) -> None:
        result = compute_word_distance(
            WordDistanceRequest(word1=(1, 0, 1, 0), word2=(0, 1, 0, 1))
        )
        assert result.distance == 4
        assert result.differing_coordinates == (0, 1, 2, 3)
        assert result.support_intersection == 0

    def test_partial_overlap(self) -> None:
        result = compute_word_distance(
            WordDistanceRequest(word1=(1, 1, 0, 0), word2=(1, 0, 1, 0))
        )
        assert result.distance == 2
        assert result.support_intersection == 1


class TestExplicitProfile:
    def test_simple_code(self) -> None:
        result = compute_explicit_profile(
            ExplicitProfileRequest(codewords=((0, 0, 0), (1, 1, 0), (0, 1, 1)))
        )
        assert result.length == 3
        assert result.cardinality == 3
        assert result.minimum_distance == 2
        assert result.maximum_distance == 2

    def test_single_codeword(self) -> None:
        # Singleton codes have no pairwise distances and are now rejected.
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ExplicitProfileRequest(codewords=((1, 0, 1),))

    def test_distance_histogram(self) -> None:
        result = compute_explicit_profile(
            ExplicitProfileRequest(codewords=((0, 0), (1, 1), (0, 1)))
        )
        # d((0,0),(1,1))=2, d((0,0),(0,1))=1, d((1,1),(0,1))=1
        assert result.distance_histogram[1] == 2
        assert result.distance_histogram[2] == 1
        assert result.minimum_distance == 1


class TestConstantWeightProfile:
    def test_weight_2_code(self) -> None:
        result = compute_constant_weight_profile(
            ConstantWeightProfileRequest(codewords=((1, 1, 0, 0), (1, 0, 1, 0)))
        )
        # intersection = 1, distance = 2*(2-1) = 2
        assert result.minimum_distance == 2
        assert result.weight == 2

    def test_disjoint_supports(self) -> None:
        result = compute_constant_weight_profile(
            ConstantWeightProfileRequest(codewords=((1, 1, 0, 0), (0, 0, 1, 1)))
        )
        # intersection = 0, distance = 2*(2-0) = 4
        assert result.minimum_distance == 4


class TestToSetSystem:
    def test_supports(self) -> None:
        result = compute_to_set_system(
            ToSetSystemRequest(codewords=((1, 0, 1, 0), (0, 1, 0, 1)))
        )
        assert result.supports == ((0, 2), (1, 3))

    def test_zero_word(self) -> None:
        result = compute_to_set_system(
            ToSetSystemRequest(codewords=((0, 0, 0), (1, 0, 1)))
        )
        assert result.supports == ((), (0, 2))


class TestDistanceProfile:
    def test_simple_code(self) -> None:
        from jacobian.math.code_nonlinear._models import BinaryCodeRequest

        result = compute_distance_profile(
            BinaryCodeRequest(codewords=((0, 0, 0), (1, 1, 0), (0, 1, 1)))
        )
        assert result.minimum_distance == 2
        assert result.weight_profile == (0, 2, 2)


class TestConstantWeight:
    def test_weight_2_length_4(self) -> None:
        result = compute_constant_weight(ConstantWeightRequest(length=4, weight=2))
        assert result.count == 6  # C(4,2) = 6
        assert len(result.codewords) == 6
