"""Tests for ordered-difference profile operations."""

import pytest
from pydantic import ValidationError

from jacobian.canonical import CanonicalLimits, canonicalize_json
from jacobian.math.combinatorics.additive._models import (
    _MAX_DIMENSION,
    _MAX_PROFILE_RESULT_BUDGET_BYTES,
    _MAX_VECTOR_SET_SIZE,
    IntegerVector,
    OrderedDifferenceProfileRequest,
    OrderedDifferenceProfileResult,
)
from jacobian.math.combinatorics.additive._tools import TOOLS
from jacobian.math.combinatorics.additive.operations import (
    ordered_difference_profile,
)


def _vec(*coords: int) -> IntegerVector:
    return IntegerVector.model_validate({"coordinates": [str(c) for c in coords]})


def _request(*vectors: tuple[int, ...]) -> OrderedDifferenceProfileRequest:
    return OrderedDifferenceProfileRequest.model_validate(
        {
            "vectors": {
                "vectors": [{"coordinates": [str(c) for c in vec]} for vec in vectors]
            }
        }
    )


def _run_ordered(
    request: OrderedDifferenceProfileRequest,
) -> OrderedDifferenceProfileResult:
    return ordered_difference_profile(request.vectors)


class TestOrderedDifferenceProfile:
    def test_three_points_2d(self) -> None:
        """Three points in Z^2 with known differences."""
        req = _request((0, 0), (1, 0), (0, 1))
        result = _run_ordered(req)
        assert result.dimension == 2
        assert result.set_size == 3
        assert result.total_ordered_pairs == 6  # 3*2
        assert result.support_size > 0
        for entry in result.entries:
            assert entry.multiplicity == len(entry.pairs)

    def test_small_set_is_admitted_in_nine_dimensions(self) -> None:
        zero = (0,) * 9
        first = (1,) + (0,) * 8
        second = (0, 1) + (0,) * 7

        result = _run_ordered(_request(zero, first, second))

        assert result.dimension == 9
        assert result.total_ordered_pairs == 6
        assert result.support_size == 6

    def test_low_cardinality_profile_admits_parser_scale_dimension(self) -> None:
        dimension = 1_024
        result = _run_ordered(_request((0,) * dimension, (1,) + (0,) * (dimension - 1)))

        assert result.dimension == dimension
        assert result.total_ordered_pairs == 2

    def test_pair_coordinate_work_rejects_before_expansion(self) -> None:
        dimension = 1_024
        request = _request(*((index,) + (0,) * (dimension - 1) for index in range(33)))

        with pytest.raises(ValueError, match="1,000,000-coordinate work budget"):
            _run_ordered(request)

    def test_no_repeated(self) -> None:
        """A Sidon set has no repeated differences."""
        req = _request((0, 0, 0), (1, 0, 0), (0, 1, 0))
        result = _run_ordered(req)
        for entry in result.entries:
            for pair in entry.pairs:
                assert pair.left_index != pair.right_index

    def test_repeated_difference(self) -> None:
        """Four points forming a parallelogram have repeated differences."""
        req = _request((0, 0), (1, 0), (0, 1), (1, 1))
        result = _run_ordered(req)
        assert result.has_repeated_difference
        assert result.first_collision is not None
        assert result.max_multiplicity >= 2

    def test_total_pairs_formula(self) -> None:
        """Total ordered pairs must equal |A|(|A|-1)."""
        req = _request((0, 0), (1, 0), (2, 0), (3, 0))
        result = _run_ordered(req)
        assert result.total_ordered_pairs == 4 * 3

    def test_single_point(self) -> None:
        """A single point has no differences."""
        req = _request((1, 2))
        result = _run_ordered(req)
        assert result.total_ordered_pairs == 0
        assert result.support_size == 0
        assert result.entries == ()
        assert not result.has_repeated_difference

    def test_one_dimensional(self) -> None:
        """One-dimensional vectors work correctly."""
        req = _request((0,), (1,), (2,))
        result = _run_ordered(req)
        assert result.dimension == 1
        assert result.total_ordered_pairs == 6  # 3*2

    def test_mismatched_dimensions_rejected(self) -> None:
        """Vectors with different dimensions should raise."""
        with pytest.raises(ValidationError):
            OrderedDifferenceProfileRequest.model_validate(
                {
                    "vectors": {
                        "vectors": [
                            {"coordinates": ["0", "0"]},
                            {"coordinates": ["1"]},
                        ]
                    }
                }
            )

    def test_translation_invariance(self) -> None:
        base = _run_ordered(_request((0, 0), (1, 0), (0, 1)))
        shifted = _run_ordered(_request((5, -3), (6, -3), (5, -2)))
        base_diffs = {e.difference.as_int_tuple(): e.multiplicity for e in base.entries}
        shifted_diffs = {
            e.difference.as_int_tuple(): e.multiplicity for e in shifted.entries
        }
        assert base_diffs == shifted_diffs

    def test_sign_reversal_symmetry(self) -> None:
        result = _run_ordered(_request((0, 0), (1, 0), (0, 1), (1, 1)))
        mult = {e.difference.as_int_tuple(): e.multiplicity for e in result.entries}
        assert mult
        for d, count in mult.items():
            assert mult[tuple(-c for c in d)] == count

    def test_result_retains_canonical_source(self) -> None:
        req = _request((0, 0), (1, 0), (0, 1))
        result = _run_ordered(req)
        assert result.vectors == req.vectors

    def test_retained_source_feeds_requests_unchanged(self) -> None:
        """The canonical IntegerVectorSet result value composes: it can be
        supplied unchanged as the source of another vector-set request."""
        result = _run_ordered(_request((0, 0), (1, 0), (0, 1)))
        recomputed = _run_ordered(
            OrderedDifferenceProfileRequest(vectors=result.vectors)
        )
        assert recomputed == result

    def test_result_rejects_empty_source(self) -> None:
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

    def test_result_replays_every_difference_from_source(self) -> None:
        req = _request((0, 0), (1, 0), (1, 1), (0, 1))
        result = _run_ordered(req)
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

    def test_result_rejects_later_collision_as_first_witness(self) -> None:
        """The witness must be pairs[0] of the first sorted repeated entry,
        not a designated pair from any later repeated entry."""
        req = _request((0, 0), (1, 0), (0, 1), (1, 1))
        result = _run_ordered(req)
        payload = result.model_dump(mode="json")
        repeated = [e for e in payload["entries"] if e["multiplicity"] > 1]
        assert len(repeated) >= 2
        payload["first_collision"] = repeated[-1]["pairs"][0]
        with pytest.raises(ValidationError):
            OrderedDifferenceProfileResult.model_validate(payload)

    def test_result_rejects_nondesignated_pair_from_first_entry(self) -> None:
        """Swapping in a different valid pair of the same first repeated
        entry breaks the canonical lexicographic pair order, so the swap
        cannot move a forged pair into the witness position."""
        req = _request((0, 0), (1, 0), (0, 1), (1, 1))
        result = _run_ordered(req)
        payload = result.model_dump(mode="json")
        first_repeated = next(e for e in payload["entries"] if e["multiplicity"] > 1)
        assert len(first_repeated["pairs"]) >= 2
        first_repeated["pairs"][0], first_repeated["pairs"][1] = (
            first_repeated["pairs"][1],
            first_repeated["pairs"][0],
        )
        with pytest.raises(ValidationError):
            OrderedDifferenceProfileResult.model_validate(payload)

    def test_first_collision_is_canonical_minimum_pair(self) -> None:
        """The witness must equal the lexicographic minimum pair of the
        first sorted repeated-difference entry."""
        req = _request((0, 0), (1, 0), (0, 1), (1, 1), (3, 2))
        result = _run_ordered(req)
        assert result.has_repeated_difference
        first_repeated = next(e for e in result.entries if e.multiplicity > 1)
        minimum = min((p.left_index, p.right_index) for p in first_repeated.pairs)
        assert result.first_collision is not None
        assert (
            result.first_collision.left_index,
            result.first_collision.right_index,
        ) == minimum

    def test_request_coordinate_bound_enforced_before_integer_conversion(self) -> None:
        """A seven-digit coordinate must fail the domain digit bound on the
        string itself; a coordinate longer than any legal vector string is
        rejected by the schema-level character ceiling without parsing."""
        seven_digits = "1234567"
        with pytest.raises(ValidationError):
            OrderedDifferenceProfileRequest.model_validate(
                {
                    "vectors": {
                        "vectors": [
                            {"coordinates": [seven_digits]},
                            {"coordinates": ["0"]},
                        ]
                    }
                }
            )
        with pytest.raises(ValidationError) as schema_error:
            IntegerVector.model_validate({"coordinates": ["9" * 100_000]})
        assert schema_error.value.errors()[0]["type"] == "string_too_long"

    def test_difference_coordinates_may_carry_one_extra_digit(self) -> None:
        """Exact differences of maximally bounded sources stay representable."""
        req = _request((-999999,), (999999,))
        result = _run_ordered(req)
        diffs = {e.difference.as_int_tuple()[0] for e in result.entries}
        assert diffs == {-1999998, 1999998}

    def test_request_schema_exposes_vector_constraints(self) -> None:
        """math.find consumers must see the set-size and coordinate ceilings."""
        request_schema = OrderedDifferenceProfileRequest.model_json_schema()
        vector_set_schema = request_schema["$defs"]["IntegerVectorSet"]
        assert vector_set_schema["properties"]["vectors"]["maxItems"] == (
            _MAX_VECTOR_SET_SIZE
        )
        vector_schema = request_schema["$defs"]["IntegerVector"]
        assert vector_schema["properties"]["coordinates"]["items"]["maxLength"] == 8

    def test_published_dimension_limit_matches_parser_ceiling(self) -> None:
        """Discovery metadata must advertise the widened 1..1024 dimension range."""
        request_schema = OrderedDifferenceProfileRequest.model_json_schema()
        vectors_description = request_schema["properties"]["vectors"]["description"]
        assert f"1<=d<={_MAX_DIMENSION}" in vectors_description
        assert "1<=d<=8" not in vectors_description

        operation = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "additive.ordered_difference_profile.compute"
        )
        assert f"1<=d<={_MAX_DIMENSION}" in operation.description
        assert "1<=d<=8" not in operation.description
        example_description = operation.examples[0].description
        assert f"1..{_MAX_DIMENSION}" in example_description
        assert "1..8" not in example_description

    def test_set_size_above_derived_bound_rejected(self) -> None:
        vectors = [
            {"coordinates": [str(i), str(i * i)]}
            for i in range(_MAX_VECTOR_SET_SIZE + 1)
        ]
        with pytest.raises(ValidationError):
            OrderedDifferenceProfileRequest.model_validate(
                {"vectors": {"vectors": vectors}}
            )

    @pytest.mark.scale
    def test_worst_case_profile_stays_within_result_budget(self) -> None:
        """A maximal Sidon family has n*(n-1) distinct difference entries,
        which is the worst-case serialized shape for the bound."""
        side = _MAX_VECTOR_SET_SIZE
        vectors = [
            (i, i * i, i * i, i * i, i * i, i * i, i * i, i * i) for i in range(side)
        ]
        req = _request(*vectors)
        result = _run_ordered(req)
        assert result.support_size == side * (side - 1)
        payload = result.model_dump(mode="json")
        encoded = canonicalize_json(
            payload,
            limits=CanonicalLimits(max_output_bytes=_MAX_PROFILE_RESULT_BUDGET_BYTES),
        )
        assert len(encoded) <= _MAX_PROFILE_RESULT_BUDGET_BYTES

    def test_result_roundtrip(self) -> None:
        req = _request((0, 0), (1, 0), (0, 1), (1, 1))
        result = _run_ordered(req)
        assert (
            OrderedDifferenceProfileResult.model_validate(result.model_dump()) == result
        )
