"""Tests for additive combinatorics operations."""

import pytest
from pydantic import ValidationError

from jacobian.canonical import parse_canonical_integer
from jacobian.math.additive_combinatorics._models import (
    AdditiveEnergyRequest,
    DirectSumPredicateRequest,
    FiniteIntegerSet,
    IntegerVector,
    IntegerVectorSet,
    OrderedDifferenceClass,
    OrderedDifferencePair,
    OrderedDifferenceProfileRequest,
    OrderedDifferenceProfileResult,
    RepresentationProfileRequest,
    SumsetCardinalityRequest,
)
from jacobian.math.additive_combinatorics._operations import (
    compute_additive_energy,
    compute_ordered_difference_profile,
    compute_representation_profile,
    compute_sumset_cardinality,
    decide_direct_sum_predicate,
)


class TestRepresentationProfile:
    def test_two_by_two(self):
        req = RepresentationProfileRequest(
            left=FiniteIntegerSet(elements=("1", "2")),
            right=FiniteIntegerSet(elements=("3", "4")),
        )
        result = compute_representation_profile(req)
        entries = {e.sum: e.multiplicity for e in result.entries}
        assert entries == {"4": 1, "5": 2, "6": 1}

    def test_empty_set(self):
        req = RepresentationProfileRequest(
            left=FiniteIntegerSet(elements=()),
            right=FiniteIntegerSet(elements=("1", "2")),
        )
        result = compute_representation_profile(req)
        assert result.entries == ()

    def test_self_sum(self):
        req = RepresentationProfileRequest(
            left=FiniteIntegerSet(elements=("0", "1", "2")),
            right=FiniteIntegerSet(elements=("0", "1", "2")),
        )
        result = compute_representation_profile(req)
        entries = {e.sum: e.multiplicity for e in result.entries}
        assert entries == {"0": 1, "1": 2, "2": 3, "3": 2, "4": 1}

    def test_negative_integers(self):
        req = RepresentationProfileRequest(
            left=FiniteIntegerSet(elements=("-2", "-1")),
            right=FiniteIntegerSet(elements=("3", "4")),
        )
        result = compute_representation_profile(req)
        entries = tuple((entry.sum, entry.multiplicity) for entry in result.entries)
        assert entries == (("1", 1), ("2", 2), ("3", 1))

    def test_sums_sorted_and_unique(self):
        req = RepresentationProfileRequest(
            left=FiniteIntegerSet(elements=("7", "-2", "0")),
            right=FiniteIntegerSet(elements=("5", "0", "-5")),
        )
        result = compute_representation_profile(req)
        assert tuple(entry.sum for entry in result.entries) == (
            "-7",
            "-5",
            "-2",
            "0",
            "2",
            "3",
            "5",
            "7",
            "12",
        )

        assert tuple(entry.sum for entry in result.entries) == tuple(
            sorted({e.sum for e in result.entries}, key=parse_canonical_integer)
        )


class TestAdditiveEnergy:
    def test_two_by_two(self):
        req = AdditiveEnergyRequest(
            left=FiniteIntegerSet(elements=("1", "2")),
            right=FiniteIntegerSet(elements=("3", "4")),
        )
        result = compute_additive_energy(req)
        assert result.energy == 6  # 1^2 + 2^2 + 1^2

    def test_equal_sets(self):
        req = AdditiveEnergyRequest(
            left=FiniteIntegerSet(elements=("0", "1")),
            right=FiniteIntegerSet(elements=("0", "1")),
        )
        result = compute_additive_energy(req)
        # A+A = {0,1,2}, r(0)=1, r(1)=2, r(2)=1 => E = 1+4+1 = 6
        assert result.energy == 6


class TestSumsetCardinality:
    def test_three_plus_two(self):
        req = SumsetCardinalityRequest(
            left=FiniteIntegerSet(elements=("0", "1", "2")),
            right=FiniteIntegerSet(elements=("0", "2")),
        )
        result = compute_sumset_cardinality(req)
        assert result.cardinality == 5
        assert result.support == ("0", "1", "2", "3", "4")

    def test_disjoint(self):
        req = SumsetCardinalityRequest(
            left=FiniteIntegerSet(elements=("10",)),
            right=FiniteIntegerSet(elements=("20",)),
        )
        result = compute_sumset_cardinality(req)
        assert result.cardinality == 1

    def test_sumset_support_matches_profile(self):
        req = SumsetCardinalityRequest(
            left=FiniteIntegerSet(elements=("7", "-2", "0")),
            right=FiniteIntegerSet(elements=("5", "0", "-5")),
        )
        result = compute_sumset_cardinality(req)
        assert result.support == (
            "-7",
            "-5",
            "-2",
            "0",
            "2",
            "3",
            "5",
            "7",
            "12",
        )


class TestDirectSumPredicate:
    def test_tiling_z4(self):
        req = DirectSumPredicateRequest(
            modulus=4,
            left=FiniteIntegerSet(elements=("0", "1")),
            right=FiniteIntegerSet(elements=("0", "2")),
        )
        result = decide_direct_sum_predicate(req)
        assert result.holds is True
        assert result.collisions == ()
        assert result.missing == ()

    def test_non_tiling_z4(self):
        req = DirectSumPredicateRequest(
            modulus=4,
            left=FiniteIntegerSet(elements=("0", "1")),
            right=FiniteIntegerSet(elements=("0", "1")),
        )
        result = decide_direct_sum_predicate(req)
        assert result.holds is False

    def test_z6_tiling(self):
        req = DirectSumPredicateRequest(
            modulus=6,
            left=FiniteIntegerSet(elements=("0", "1", "2")),
            right=FiniteIntegerSet(elements=("0", "3")),
        )
        result = decide_direct_sum_predicate(req)
        assert result.holds is True

    def test_empty_sets_in_z12_return_numeric_missing(self):
        req = DirectSumPredicateRequest(
            modulus=12,
            left=FiniteIntegerSet(elements=()),
            right=FiniteIntegerSet(elements=()),
        )
        result = decide_direct_sum_predicate(req)
        assert result.holds is False
        assert result.missing == tuple(str(value) for value in range(12))


class TestOrderedDifferenceProfile:
    def _req(self, vecs):
        return OrderedDifferenceProfileRequest(
            vectors=IntegerVectorSet(
                vectors=tuple(
                    IntegerVector(coordinates=tuple(str(c) for c in v)) for v in vecs
                ),
            ),
        )

    def test_rectangle_repeated_difference(self):
        request = self._req([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = compute_ordered_difference_profile(request)
        assert result.vectors == request.vectors
        assert result.set_size == 4
        assert result.ordered_pair_count == 12  # 4 * 3
        assert result.has_repeated_difference
        assert result.max_multiplicity == 2
        assert result.first_repeated_difference == ("-1", "0")
        # The difference (1,0) is realized by two ordered pairs.
        diff_10 = [
            c for c in result.classes if tuple(c.difference.coordinates) == ("1", "0")
        ]
        assert len(diff_10) == 1
        assert len(diff_10[0].pairs) == 2
        assert {p.minuend_index for p in diff_10[0].pairs} == {1, 2}

    def test_triangle_is_sidon(self):
        result = compute_ordered_difference_profile(
            self._req([(0, 0), (1, 0), (0, 1)]),
        )
        assert result.set_size == 3
        assert result.ordered_pair_count == 6
        assert not result.has_repeated_difference
        assert result.first_repeated_difference is None
        assert result.support_size == 6  # all 6 nonzero ordered differences distinct
        assert result.max_multiplicity == 1

    def test_one_dimension_agrees_with_sidon(self):
        # A 1D Sidon set {0,1,3} has all ordered differences distinct.
        result = compute_ordered_difference_profile(
            self._req([(0,), (1,), (3,)]),
        )
        assert result.dimension == 1
        assert result.ordered_pair_count == 6
        assert not result.has_repeated_difference
        diffs = {tuple(c.difference.coordinates) for c in result.classes}
        assert diffs == {("1",), ("3",), ("2",), ("-1",), ("-3",), ("-2",)}

    def test_translation_invariance(self):
        base = compute_ordered_difference_profile(
            self._req([(0, 0), (1, 0), (0, 1)]),
        )
        shifted = compute_ordered_difference_profile(
            self._req([(5, -3), (6, -3), (5, -2)]),
        )
        base_diffs = {
            tuple(c.difference.coordinates): len(c.pairs) for c in base.classes
        }
        shifted_diffs = {
            tuple(c.difference.coordinates): len(c.pairs) for c in shifted.classes
        }
        assert base_diffs == shifted_diffs

    def test_sign_reversal(self):
        result = compute_ordered_difference_profile(
            self._req([(0, 0), (1, 0), (0, 1), (1, 1)]),
        )
        diffs = {tuple(c.difference.coordinates): len(c.pairs) for c in result.classes}
        # For every difference v, the multiplicity of -v must equal that of v.
        for d, count in diffs.items():
            neg = tuple(str(-int(x)) for x in d)
            assert diffs[neg] == count

    def test_rejects_duplicate_vectors(self):
        with pytest.raises(ValueError, match="distinct"):
            IntegerVectorSet(
                vectors=(
                    IntegerVector(coordinates=("0", "0")),
                    IntegerVector(coordinates=("0", "0")),
                ),
            )

    def test_rejects_mixed_dimensions(self):
        with pytest.raises(ValueError, match="dimension"):
            IntegerVectorSet(
                vectors=(
                    IntegerVector(coordinates=("0", "0")),
                    IntegerVector(coordinates=("0",)),
                ),
            )

    def test_result_replays_every_difference_from_source(self):
        request = self._req([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = compute_ordered_difference_profile(request)
        points = tuple(
            tuple(int(c) for c in vec.coordinates) for vec in result.vectors.vectors
        )
        seen = set()
        for cls in result.classes:
            diff = tuple(int(c) for c in cls.difference.coordinates)
            for pair in cls.pairs:
                replayed = tuple(
                    points[pair.minuend_index][k] - points[pair.subtrahend_index][k]
                    for k in range(result.dimension)
                )
                assert replayed == diff
                seen.add((pair.minuend_index, pair.subtrahend_index))
        n = result.set_size
        assert seen == {(i, j) for i in range(n) for j in range(n) if i != j}
        OrderedDifferenceProfileResult.model_validate(result.model_dump())

    def test_result_rejects_forged_difference_independent_of_source(self):
        vectors = IntegerVectorSet(
            vectors=(
                IntegerVector(coordinates=("0",)),
                IntegerVector(coordinates=("1",)),
            ),
        )
        with pytest.raises(ValidationError, match="source\\[minuend\\]"):
            OrderedDifferenceProfileResult(
                vectors=vectors,
                dimension=1,
                set_size=2,
                classes=(
                    OrderedDifferenceClass(
                        difference=IntegerVector(coordinates=("-999",)),
                        pairs=(
                            OrderedDifferencePair(minuend_index=0, subtrahend_index=1),
                        ),
                    ),
                    OrderedDifferenceClass(
                        difference=IntegerVector(coordinates=("999",)),
                        pairs=(
                            OrderedDifferencePair(minuend_index=1, subtrahend_index=0),
                        ),
                    ),
                ),
                ordered_pair_count=2,
                support_size=2,
                max_multiplicity=1,
                has_repeated_difference=False,
                first_repeated_difference=None,
            )

    def test_result_rejects_mutated_source(self):
        result = compute_ordered_difference_profile(
            self._req([(0, 0), (1, 0), (1, 1), (0, 1)]),
        )
        payload = result.model_dump()
        payload["vectors"]["vectors"][0]["coordinates"] = ["9", "9"]
        with pytest.raises(ValidationError, match="source\\[minuend\\]"):
            OrderedDifferenceProfileResult.model_validate(payload)

    def test_result_rejects_false_repeated_flag(self):
        result = compute_ordered_difference_profile(
            self._req([(0, 0), (1, 0), (1, 1), (0, 1)]),
        )
        payload = result.model_dump()
        payload["has_repeated_difference"] = False
        payload["first_repeated_difference"] = None
        with pytest.raises(ValidationError, match="max_multiplicity > 1"):
            OrderedDifferenceProfileResult.model_validate(payload)

    def test_result_rejects_wrong_repeated_witness(self):
        result = compute_ordered_difference_profile(
            self._req([(0, 0), (1, 0), (1, 1), (0, 1)]),
        )
        payload = result.model_dump()
        later = next(
            cls.difference.coordinates
            for cls in result.classes
            if len(cls.pairs) > 1
            and tuple(cls.difference.coordinates)
            != tuple(result.first_repeated_difference or ())
        )
        payload["first_repeated_difference"] = list(later)
        with pytest.raises(ValidationError, match="first class of multiplicity"):
            OrderedDifferenceProfileResult.model_validate(payload)

    def test_request_schema_publishes_coordinate_digit_bound(self):
        schema = OrderedDifferenceProfileRequest.model_json_schema()
        vector = schema["$defs"]["IntegerVector"]["properties"]["coordinates"]
        assert "64 decimal digits" in vector["description"]
        assert vector["items"]["pattern"] == r"^(?:0|-?[1-9][0-9]{0,63})$"
        assert vector["items"]["maxLength"] == 65
        assert "64 decimal digits" in schema["description"]

    def test_rejects_65_digit_coordinate(self):
        with pytest.raises(ValidationError):
            IntegerVector(coordinates=("1" * 65,))

    def test_accepts_64_digit_coordinate(self):
        vector = IntegerVector(coordinates=("-" + "1" * 64,))
        assert vector.coordinates == ("-" + "1" * 64,)
