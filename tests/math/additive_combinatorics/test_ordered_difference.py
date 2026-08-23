"""Tests for ordered-difference profile operations."""

import pytest
from pydantic import ValidationError

from jacobian.canonical import CanonicalLimits, canonicalize_json
from jacobian.math.additive_combinatorics._models import (
    _MAX_PROFILE_RESULT_BUDGET_BYTES,
    _MAX_VECTOR_SET_SIZE,
    IntegerVector,
    OrderedDifferenceProfileRequest,
    OrderedDifferenceProfileResult,
)
from jacobian.math.additive_combinatorics._operations import (
    compute_ordered_difference_profile,
)


def _vec(*coords: int) -> IntegerVector:
    return IntegerVector.model_validate({"coordinates": [str(c) for c in coords]})


def _request(*vectors: tuple[int, ...]) -> OrderedDifferenceProfileRequest:
    return OrderedDifferenceProfileRequest(
        vectors={"vectors": [{"coordinates": [str(c) for c in vec]} for vec in vectors]}
    )


class TestOrderedDifferenceProfile:
    def test_three_points_2d(self):
        """Three points in Z^2 with known differences."""
        req = _request((0, 0), (1, 0), (0, 1))
        result = compute_ordered_difference_profile(req)
        assert result.dimension == 2
        assert result.set_size == 3
        assert result.total_ordered_pairs == 6  # 3*2
        assert result.support_size > 0
        for entry in result.entries:
            assert entry.multiplicity == len(entry.pairs)

    def test_no_repeated(self):
        """A Sidon set has no repeated differences."""
        req = _request((0, 0, 0), (1, 0, 0), (0, 1, 0))
        result = compute_ordered_difference_profile(req)
        for entry in result.entries:
            for pair in entry.pairs:
                assert pair.left_index != pair.right_index

    def test_repeated_difference(self):
        """Four points forming a parallelogram have repeated differences."""
        req = _request((0, 0), (1, 0), (0, 1), (1, 1))
        result = compute_ordered_difference_profile(req)
        assert result.has_repeated_difference
        assert result.first_collision is not None
        assert result.max_multiplicity >= 2

    def test_total_pairs_formula(self):
        """Total ordered pairs must equal |A|(|A|-1)."""
        req = _request((0, 0), (1, 0), (2, 0), (3, 0))
        result = compute_ordered_difference_profile(req)
        assert result.total_ordered_pairs == 4 * 3

    def test_single_point(self):
        """A single point has no differences."""
        req = _request((1, 2))
        result = compute_ordered_difference_profile(req)
        assert result.total_ordered_pairs == 0
        assert result.support_size == 0
        assert result.entries == ()
        assert not result.has_repeated_difference

    def test_one_dimensional(self):
        """One-dimensional vectors work correctly."""
        req = _request((0,), (1,), (2,))
        result = compute_ordered_difference_profile(req)
        assert result.dimension == 1
        assert result.total_ordered_pairs == 6  # 3*2

    def test_mismatched_dimensions_rejected(self):
        """Vectors with different dimensions should raise."""
        with pytest.raises(ValidationError):
            OrderedDifferenceProfileRequest(
                vectors={
                    "vectors": [
                        {"coordinates": ["0", "0"]},
                        {"coordinates": ["1"]},
                    ]
                }
            )

    def test_translation_invariance(self):
        base = compute_ordered_difference_profile(_request((0, 0), (1, 0), (0, 1)))
        shifted = compute_ordered_difference_profile(
            _request((5, -3), (6, -3), (5, -2))
        )
        base_diffs = {e.difference.as_int_tuple(): e.multiplicity for e in base.entries}
        shifted_diffs = {
            e.difference.as_int_tuple(): e.multiplicity for e in shifted.entries
        }
        assert base_diffs == shifted_diffs

    def test_sign_reversal_symmetry(self):
        result = compute_ordered_difference_profile(
            _request((0, 0), (1, 0), (0, 1), (1, 1))
        )
        mult = {e.difference.as_int_tuple(): e.multiplicity for e in result.entries}
        assert mult
        for d, count in mult.items():
            assert mult[tuple(-c for c in d)] == count

    def test_result_retains_canonical_source(self):
        req = _request((0, 0), (1, 0), (0, 1))
        result = compute_ordered_difference_profile(req)
        assert result.vectors == req.vectors

    def test_retained_source_feeds_requests_unchanged(self):
        """The canonical IntegerVectorSet result value composes: it can be
        supplied unchanged as the source of another vector-set request."""
        result = compute_ordered_difference_profile(_request((0, 0), (1, 0), (0, 1)))
        recomputed = compute_ordered_difference_profile(
            OrderedDifferenceProfileRequest(vectors=result.vectors)
        )
        assert recomputed == result

    def test_result_rejects_empty_source(self):
        """The kernel never emits a profile without a valid nonempty source,
        so an empty serialized result cannot pass the trust boundary."""
        with pytest.raises(ValidationError):
            OrderedDifferenceProfileResult.model_validate(
                {
                    "vectors": {"vectors": []},
                    "dimension": 1,
                    "set_size": 0,
                    "total_ordered_pairs": 0,
                    "support_size": 0,
                    "max_multiplicity": 0,
                }
            )

    def test_result_replays_every_difference_from_source(self):
        req = _request((0, 0), (1, 0), (1, 1), (0, 1))
        result = compute_ordered_difference_profile(req)
        seen = set()
        source = [v.as_int_tuple() for v in result.vectors.vectors]
        for entry in result.entries:
            difference = entry.difference.as_int_tuple()
            for pair in entry.pairs:
                replayed = tuple(
                    source[pair.left_index][k] - source[pair.right_index][k]
                    for k in range(result.dimension)
                )
                assert replayed == difference
                seen.add((pair.left_index, pair.right_index))
        n = result.set_size
        assert seen == {(i, j) for i in range(n) for j in range(n) if i != j}

    def test_result_rejects_forged_difference(self):
        req = _request((0,), (1,))
        result = compute_ordered_difference_profile(req)
        payload = result.model_dump()
        payload["entries"][-1]["difference"] = {"coordinates": ["2"]}
        with pytest.raises(ValidationError, match="pair difference must match vectors"):
            OrderedDifferenceProfileResult.model_validate(payload)

    def test_result_rejects_mutated_source(self):
        req = _request((0, 0), (1, 0), (1, 1), (0, 1))
        result = compute_ordered_difference_profile(req)
        payload = result.model_dump()
        payload["vectors"]["vectors"] = (
            {"coordinates": ["9", "9"]},
            *payload["vectors"]["vectors"][1:],
        )
        with pytest.raises(ValidationError, match="pair difference must match vectors"):
            OrderedDifferenceProfileResult.model_validate(payload)

    def test_result_rejects_later_collision_as_first_witness(self):
        """The witness must be pairs[0] of the first sorted repeated entry,
        not a designated pair from any later repeated entry."""
        req = _request((0, 0), (1, 0), (0, 1), (1, 1))
        result = compute_ordered_difference_profile(req)
        payload = result.model_dump(mode="json")
        repeated = [e for e in payload["entries"] if e["multiplicity"] > 1]
        assert len(repeated) >= 2
        payload["first_collision"] = repeated[-1]["pairs"][0]
        with pytest.raises(ValidationError, match="designated pair"):
            OrderedDifferenceProfileResult.model_validate(payload)

    def test_result_rejects_nondesignated_pair_from_first_entry(self):
        """Swapping in a different valid pair of the same first repeated
        entry breaks the canonical lexicographic pair order, so the swap
        cannot move a forged pair into the witness position."""
        req = _request((0, 0), (1, 0), (0, 1), (1, 1))
        result = compute_ordered_difference_profile(req)
        payload = result.model_dump(mode="json")
        first_repeated = next(e for e in payload["entries"] if e["multiplicity"] > 1)
        assert len(first_repeated["pairs"]) >= 2
        first_repeated["pairs"][0], first_repeated["pairs"][1] = (
            first_repeated["pairs"][1],
            first_repeated["pairs"][0],
        )
        with pytest.raises(ValidationError, match="lexicographic order"):
            OrderedDifferenceProfileResult.model_validate(payload)

    def test_first_collision_is_canonical_minimum_pair(self):
        """The witness must equal the lexicographic minimum pair of the
        first sorted repeated-difference entry."""
        req = _request((0, 0), (1, 0), (0, 1), (1, 1), (3, 2))
        result = compute_ordered_difference_profile(req)
        assert result.has_repeated_difference
        first_repeated = next(e for e in result.entries if e.multiplicity > 1)
        minimum = min((p.left_index, p.right_index) for p in first_repeated.pairs)
        assert result.first_collision is not None
        assert (
            result.first_collision.left_index,
            result.first_collision.right_index,
        ) == minimum

    def test_request_coordinate_bound_enforced_before_integer_conversion(self):
        """A seven-digit coordinate must fail the domain digit bound on the
        string itself; a coordinate longer than any legal vector string is
        rejected by the schema-level character ceiling without parsing."""
        seven_digits = "1234567"
        with pytest.raises(
            ValidationError, match="exceeds the 6-digit coordinate bound"
        ):
            OrderedDifferenceProfileRequest(
                vectors={
                    "vectors": [
                        {"coordinates": [seven_digits]},
                        {"coordinates": ["0"]},
                    ]
                }
            )
        with pytest.raises(ValidationError) as schema_error:
            IntegerVector(coordinates=["9" * 100_000])
        assert schema_error.value.errors()[0]["type"] == "string_too_long"

    def test_difference_coordinates_may_carry_one_extra_digit(self):
        """Exact differences of maximally bounded sources stay representable."""
        req = _request((-999999,), (999999,))
        result = compute_ordered_difference_profile(req)
        diffs = {e.difference.as_int_tuple()[0] for e in result.entries}
        assert diffs == {-1999998, 1999998}

    def test_request_schema_exposes_vector_constraints(self):
        """math.find consumers must see the set-size and coordinate ceilings."""
        request_schema = OrderedDifferenceProfileRequest.model_json_schema()
        vector_set_schema = request_schema["$defs"]["IntegerVectorSet"]
        assert vector_set_schema["properties"]["vectors"]["maxItems"] == (
            _MAX_VECTOR_SET_SIZE
        )
        vector_schema = request_schema["$defs"]["IntegerVector"]
        assert vector_schema["properties"]["coordinates"]["items"]["maxLength"] == 8

    def test_set_size_above_derived_bound_rejected(self):
        vectors = [
            {"coordinates": [str(i), str(i * i)]}
            for i in range(_MAX_VECTOR_SET_SIZE + 1)
        ]
        with pytest.raises(ValidationError):
            OrderedDifferenceProfileRequest(vectors={"vectors": vectors})

    def test_worst_case_profile_stays_within_result_budget(self):
        """A maximal Sidon family has n*(n-1) distinct difference entries,
        which is the worst-case serialized shape for the bound."""
        side = _MAX_VECTOR_SET_SIZE
        vectors = [
            (i, i * i, i * i, i * i, i * i, i * i, i * i, i * i) for i in range(side)
        ]
        req = _request(*vectors)
        result = compute_ordered_difference_profile(req)
        assert result.support_size == side * (side - 1)
        payload = result.model_dump(mode="json")
        encoded = canonicalize_json(
            payload,
            limits=CanonicalLimits(max_output_bytes=_MAX_PROFILE_RESULT_BUDGET_BYTES),
        )
        assert len(encoded) <= _MAX_PROFILE_RESULT_BUDGET_BYTES

    def test_result_roundtrip(self):
        req = _request((0, 0), (1, 0), (0, 1), (1, 1))
        result = compute_ordered_difference_profile(req)
        assert (
            OrderedDifferenceProfileResult.model_validate(result.model_dump()) == result
        )
