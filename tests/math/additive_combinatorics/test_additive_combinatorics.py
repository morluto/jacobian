"""Tests for additive combinatorics operations."""

import pytest
from pydantic import ValidationError

from jacobian.canonical import parse_canonical_integer
from jacobian.math.additive_combinatorics._models import (
    AdditiveEnergyRequest,
    DifferenceClassEntry,
    DirectSumPredicateRequest,
    FiniteIntegerSet,
    FiniteIntegerVectorSet,
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
            vectors=FiniteIntegerVectorSet(vectors=tuple(tuple(v) for v in vecs)),
        )

    def test_rectangle_repeated_difference(self):
        request = self._req([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = compute_ordered_difference_profile(request)
        assert result.source_set == request.vectors
        assert result.set_size == 4
        assert result.total_ordered_pairs == 12  # 4 * 3
        assert result.has_repeated_difference
        assert result.max_multiplicity == 2
        assert result.first_repeated_difference == (-1, 0)
        # The difference (1,0) is realized by two ordered pairs.
        diff_10 = [
            c for c in result.classes if c.difference == (1, 0)
        ]
        assert len(diff_10) == 1
        assert len(diff_10[0].source_pairs) == 2
        assert {a for a, _ in diff_10[0].source_pairs} == {1, 2}

    def test_triangle_is_sidon(self):
        result = compute_ordered_difference_profile(
            self._req([(0, 0), (1, 0), (0, 1)]),
        )
        assert result.set_size == 3
        assert result.total_ordered_pairs == 6
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
        assert result.total_ordered_pairs == 6
        assert not result.has_repeated_difference
        diffs = {c.difference for c in result.classes}
        assert diffs == {(1,), (3,), (2,), (-1,), (-3,), (-2,)}

    def test_translation_invariance(self):
        base = compute_ordered_difference_profile(
            self._req([(0, 0), (1, 0), (0, 1)]),
        )
        shifted = compute_ordered_difference_profile(
            self._req([(5, -3), (6, -3), (5, -2)]),
        )
        base_diffs = {
            c.difference: len(c.source_pairs) for c in base.classes
        }
        shifted_diffs = {
            c.difference: len(c.source_pairs) for c in shifted.classes
        }
        assert base_diffs == shifted_diffs

    def test_sign_reversal(self):
        result = compute_ordered_difference_profile(
            self._req([(0, 0), (1, 0), (0, 1), (1, 1)]),
        )
        diffs = {
            c.difference: len(c.source_pairs) for c in result.classes
        }
        # For every difference v, the multiplicity of -v must equal that of v.
        for d, count in diffs.items():
            neg = tuple(-x for x in d)
            assert diffs[neg] == count

    def test_rejects_duplicate_vectors(self):
        with pytest.raises(ValueError, match="distinct"):
            FiniteIntegerVectorSet(
                vectors=((0, 0), (0, 0)),
            )

    def test_rejects_mixed_dimensions(self):
        with pytest.raises(ValueError, match="dimension"):
            FiniteIntegerVectorSet(
                vectors=((0, 0), (0,)),
            )

    def test_result_replays_every_difference_from_source(self):
        request = self._req([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = compute_ordered_difference_profile(request)
        points = result.source_set.vectors
        seen = set()
        for cls in result.classes:
            diff = cls.difference
            for minuend, subtrahend in cls.source_pairs:
                replayed = tuple(
                    points[minuend][k] - points[subtrahend][k]
                    for k in range(result.dimension)
                )
                assert replayed == diff
                seen.add((minuend, subtrahend))
        n = result.set_size
        assert seen == {(i, j) for i in range(n) for j in range(n) if i != j}
        OrderedDifferenceProfileResult.model_validate(result.model_dump())

    def test_result_rejects_forged_difference_independent_of_source(self):
        vectors = FiniteIntegerVectorSet(vectors=((0,), (1,)))
        with pytest.raises(ValidationError, match="replay"):
            OrderedDifferenceProfileResult(
                source_set=vectors,
                dimension=1,
                set_size=2,
                total_ordered_pairs=2,
                support_size=2,
                max_multiplicity=1,
                has_repeated_difference=False,
                first_repeated_difference=None,
                classes=(
                    DifferenceClassEntry(
                        difference=(-999,),
                        multiplicity=1,
                        source_pairs=((0, 1),),
                    ),
                    DifferenceClassEntry(
                        difference=(999,),
                        multiplicity=1,
                        source_pairs=((1, 0),),
                    ),
                ),
            )

    def test_result_rejects_mutated_source(self):
        result = compute_ordered_difference_profile(
            self._req([(0, 0), (1, 0), (1, 1), (0, 1)]),
        )
        payload = result.model_dump()
        payload["source_set"]["vectors"] = [(9, 9), *payload["source_set"]["vectors"][1:]]
        with pytest.raises(ValidationError, match="replay"):
            OrderedDifferenceProfileResult.model_validate(payload)

    def test_result_rejects_false_repeated_flag(self):
        result = compute_ordered_difference_profile(
            self._req([(0, 0), (1, 0), (1, 1), (0, 1)]),
        )
        payload = result.model_dump()
        payload["has_repeated_difference"] = False
        payload["first_repeated_difference"] = None
        with pytest.raises(ValidationError, match="exact replay"):
            OrderedDifferenceProfileResult.model_validate(payload)

    def test_result_rejects_wrong_repeated_witness(self):
        result = compute_ordered_difference_profile(
            self._req([(0, 0), (1, 0), (1, 1), (0, 1)]),
        )
        payload = result.model_dump()
        later = next(
            cls.difference
            for cls in result.classes
            if len(cls.source_pairs) > 1
            and cls.difference != result.first_repeated_difference
        )
        payload["first_repeated_difference"] = list(later)
        with pytest.raises(ValidationError, match="lexicographically"):
            OrderedDifferenceProfileResult.model_validate(payload)

    def test_result_requires_witness_for_actual_repetition(self):
        result = compute_ordered_difference_profile(
            self._req([(0, 0), (1, 0), (1, 1), (0, 1)]),
        )
        payload = result.model_dump()
        payload["first_repeated_difference"] = None
        with pytest.raises(ValidationError, match="lexicographically"):
            OrderedDifferenceProfileResult.model_validate(payload)

    def test_request_schema_publishes_coordinate_digit_bound(self):
        schema = OrderedDifferenceProfileRequest.model_json_schema()
        assert "MAX_VECTOR_COORDINATE_DIGITS" in schema["$defs"][
            "FiniteIntegerVectorSet"
        ]["description"]

    def test_rejects_coordinate_beyond_digit_bound(self):
        overflow = 10**64
        with pytest.raises(ValidationError, match="digit bound"):
            FiniteIntegerVectorSet(vectors=((0,), (overflow,)))

    def test_accepts_coordinate_at_digit_bound(self):
        largest = 10**64 - 1
        vector_set = FiniteIntegerVectorSet(vectors=((0,), (largest,)))
        assert vector_set.vectors[1] == (largest,)

    def test_rejects_forged_oversized_claimed_difference(self):
        with pytest.raises(ValidationError, match="difference coordinates"):
            DifferenceClassEntry(
                difference=(10**65,),
                multiplicity=1,
                source_pairs=((0, 1),),
            )
